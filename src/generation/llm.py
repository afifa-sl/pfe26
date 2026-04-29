"""Client HuggingFace pour l'inférence LLM — optimisé Google Colab."""
import time
import logging
from typing import Optional, Generator
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HFClient:
    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-1.5B-Instruct",   # ← modèle léger par défaut
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        temperature: float = 0.0,
        max_tokens: int = 512,
        warm_up: bool = True,
    ):
        self.model_name  = model
        self.device      = device
        self.temperature = temperature
        self.max_tokens  = max_tokens

        self._check_connection()
        if warm_up:
            self._warm_up()

    def _check_connection(self) -> None:
        try:
            logger.info(f"Chargement {self.model_name} sur {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
            )
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            self.model.eval()
            logger.info(f"✓ Modèle chargé sur {self.device}")
        except OSError:
            raise ConnectionError(
                f"Modèle '{self.model_name}' introuvable.\n"
                f"  → Vérifiez le nom sur https://huggingface.co/models\n"
                f"  → Ou connectez-vous : huggingface-cli login"
            )

    def _warm_up(self) -> None:
        try:
            logger.info("Préchauffage...")
            start = time.time()
            inputs = self.tokenizer("Bonjour", return_tensors="pt").to(self.device)
            with torch.no_grad():
                self.model.generate(**inputs, max_new_tokens=3)
            logger.info(f"✓ Prêt en {time.time()-start:.2f}s")
        except Exception as e:
            logger.warning(f"⚠️  Préchauffage échoué (optionnel) : {e}")

    def _optimize_prompt(self, prompt: str, max_length: int = 12000) -> str:
        """Tronque le prompt si nécessaire pour rester dans la fenêtre de contexte.
        Limite par défaut à 12000 chars (~3000 tokens) pour laisser de la place
        aux réponses longues (listes, tableaux)."""
        if len(prompt) > max_length:
            prompt = prompt[:max_length] + "\n...\n[Suite tronquée — réponds avec les données disponibles]"
        return prompt

    def _apply_chat_template(self, prompt: str, system: Optional[str]) -> str:
        """Applique le chat template du modèle si disponible (Qwen, Llama3, etc.)."""
        try:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # Fallback si le modèle n'a pas de chat template
            prefix = f"[System]: {system}\n" if system else ""
            return f"{prefix}[User]: {prompt}\n[Assistant]:"

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        prompt = self._optimize_prompt(prompt)
        temp   = temperature if temperature is not None else self.temperature
        tokens = max_tokens  if max_tokens  is not None else self.max_tokens

        full_prompt = self._apply_chat_template(prompt, system)

        logger.info(f"Génération ({self.model_name}, max_tokens={tokens}, temp={temp})")
        start = time.time()
        try:
            inputs      = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
            input_len   = inputs["input_ids"].shape[1]   # ← longueur du prompt
            do_sample   = temp > 0

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=tokens,
                    temperature=temp      if do_sample else None,
                    do_sample=do_sample,
                    top_k=40              if do_sample else None,
                    top_p=0.9             if do_sample else None,
                    repetition_penalty=1.2,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            # ✅ FIX CRITIQUE : décoder UNIQUEMENT les nouveaux tokens
            new_tokens = outputs[0][input_len:]
            response   = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            logger.info(f"✓ Réponse en {time.time()-start:.2f}s ({len(response)} chars)")
            return response

        except torch.cuda.OutOfMemoryError:
            raise MemoryError(
                f"VRAM insuffisante pour {self.model_name}.\n"
                f"  → Réduisez max_tokens dans config.py\n"
                f"  → Ou utilisez un modèle plus petit (ex: Qwen2.5-0.5B-Instruct)"
            )
        except Exception as e:
            raise Exception(f"Erreur génération : {e}")

    def generate_stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Generator[str, None, None]:
        prompt = self._optimize_prompt(prompt)
        temp   = temperature if temperature is not None else self.temperature
        tokens = max_tokens  if max_tokens  is not None else self.max_tokens

        full_prompt = self._apply_chat_template(prompt, system)

        inputs    = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
        do_sample = temp > 0
        streamer  = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=tokens,
            temperature=temp  if do_sample else None,
            do_sample=do_sample,
            top_k=40          if do_sample else None,
            top_p=0.9         if do_sample else None,
            repetition_penalty=1.3,
            pad_token_id=self.tokenizer.eos_token_id,
            streamer=streamer,
        )
        start, token_count = time.time(), 0
        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()
        # Détection de boucle de répétition : si la même ligne apparaît 3+ fois, on arrête
        recent_lines = []
        buffer = ""
        for token in streamer:
            if token:
                token_count += 1
                buffer += token
                yield token
                # Vérifier les lignes complètes
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line_stripped = line.strip()
                    if line_stripped:
                        recent_lines.append(line_stripped)
                        # Si les 3 dernières lignes sont identiques → boucle détectée
                        if len(recent_lines) >= 3 and len(set(recent_lines[-3:])) == 1:
                            logger.warning("⚠ Boucle de répétition détectée, arrêt de la génération")
                            thread.join()
                            return
        thread.join()
        elapsed = time.time() - start
        logger.info(f"✓ Streaming: {token_count} tokens en {elapsed:.2f}s ({token_count/max(elapsed,0.1):.1f} tok/s)")

    def get_model_info(self) -> dict:
        try:
            cfg = self.model.config.to_dict()
            return {
                "model_name" : self.model_name,
                "device"     : self.device,
                "dtype"      : str(next(self.model.parameters()).dtype),
                "vocab_size" : cfg.get("vocab_size"),
                "num_layers" : cfg.get("num_hidden_layers"),
                "hidden_size": cfg.get("hidden_size"),
            }
        except Exception as e:
            return {"error": str(e)}
