import re
import numpy as np
from sentence_transformers import SentenceTransformer
import atexit
import os
import pickle


class SimpleRAG:
    def __init__(self, filename="data/skillbook.pkl", model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.vectors = []
        self.metadata = []
        if filename:
            try:
                self.load_from_file(filename)
            except FileNotFoundError:
                pass

        def cleanup_function():
            if filename:
                print("saving skillbook before closing...")
                self.save_to_file(filename)

        atexit.register(cleanup_function)  # called when python exits

    def save_to_file(self, filename: str):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, mode="wb") as fh:
            pickle.dump(
                {
                    "vectors": self.vectors,
                    "metadata": self.metadata,
                },
                fh,
            )

    def load_from_file(self, filename: str):
        with open(filename, mode="rb") as fh:
            data = pickle.load(fh)
        self.vectors = data["vectors"]
        self.metadata = data["metadata"]

    def add(self, key: str, value: str):
        vector = self.model.encode(key)
        self.vectors.append(vector)
        self.metadata.append({"key": key, "value": value})

    def query(self, key, top_k=1, min_score=0.8):
        if not self.vectors:
            return ""
        query_vec = self.model.encode(key)
        # use cosine similarity as lookup
        similarities = [np.dot(query_vec, v) for v in self.vectors]
        if top_k != -1:
            top_indices = np.argsort(similarities)[-top_k:][::-1]
        else:
            top_indices = np.argsort(similarities)[::-1]
        results = []
        for idx in top_indices:
            if float(similarities[idx]) < min_score:
                # the top_indices are sorted, so we can break early
                break
            results.append(
                {
                    "key": self.metadata[idx]["key"],
                    "value": self.metadata[idx]["value"],
                    "score": float(similarities[idx]),
                }
            )

        return results
    
    
class DoubleSimRAG(SimpleRAG):
    def __init__(self, filename="data/skillbook_double.pkl", model_name="all-MiniLM-L6-v2", l1=0.75, l2=0.25):
        super().__init__(filename=filename, model_name=model_name)
        self.lambda1 = l1 # Weight for the ACTION
        self.lambda2 = l2 # Weight for OBJECTS + ENV INFO
        
        # Regex to handle ACTION(objs) followed by optional env info
        # Group 1: Action, Group 2: Objects, Group 3: Environment info
        self.func_with_env = re.compile(r"^(\w+)\(([^)]*)\)\s*(.*)$")
        # Template version: Template Action(objs) env
        self.temp_with_env = re.compile(r"^(\w+)\s+(\w+)\(([^)]*)\)\s*(.*)$")

    def _split_components(self, key: str):
        """
        Returns (branch, action_string, object_env_string)
        """
        t_match = self.temp_with_env.match(key)
        if t_match:
            obj_env = f"{t_match.group(3)} {t_match.group(4)}".strip()
            return "TEMPLATE", t_match.group(2), obj_env
        a_match = self.func_with_env.match(key)
        if a_match:
            obj_env = f"{a_match.group(2)} {a_match.group(3)}".strip()
            return "ACTION", a_match.group(1), obj_env
        return "GENERAL", key, ""

    def add(self, key: str, value: str):
        branch, act_part, obj_env_part = self._split_components(key)
        
        # We encode both parts. If General, obj_env_part is just empty string
        vec_act = self.model.encode(act_part)
        vec_obj = self.model.encode(obj_env_part if obj_env_part else "none")
        
        self.vectors.append((vec_act, vec_obj))
        self.metadata.append({
            "key": key, 
            "value": value, 
            "branch": branch,
            "act_str": act_part,
            "obj_str": obj_env_part
        })

    def query(self, key: str, top_k=1, min_score=0.8):
        if not self.vectors:
            return []
        
        _, q_act, q_obj_env = self._split_components(key)
        qv_act = self.model.encode(q_act)
        qv_obj = self.model.encode(q_obj_env if q_obj_env else "none")
        
        similarities = []
        for v_act, v_obj in self.vectors:
            sim_act = np.dot(qv_act, v_act) / (np.linalg.norm(qv_act) * np.linalg.norm(v_act) + 1e-9)
            sim_obj = np.dot(qv_obj, v_obj) / (np.linalg.norm(qv_obj) * np.linalg.norm(v_obj) + 1e-9)
            total_sim = (self.lambda1 * sim_act) + (self.lambda2 * sim_obj)
            similarities.append(total_sim)

        indices = np.argsort(similarities)[::-1]
        results = []
        for idx in indices:
            score = float(similarities[idx])
            if score < min_score:
                break
            results.append({
                "key": self.metadata[idx]["key"],
                "value": self.metadata[idx]["value"],
                "score": score,
                "branch": self.metadata[idx]["branch"]
            })
            if top_k != -1 and len(results) >= top_k:
                break
        return results