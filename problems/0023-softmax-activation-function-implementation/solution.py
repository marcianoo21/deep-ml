import math

def softmax(scores: list[float]) -> list[float]:
    stable_scores = [score - max(scores) for score in scores]
    exp_scores = [math.exp(stable_score) for stable_score in stable_scores]
    return [exp_score / sum(exp_scores) for exp_score in exp_scores]