import numpy as np

scores = np.loadtxt('./sushi3-2016/sushi3b.5000.10.score', delimiter=' ')

print('scores.shape: {}'.format(scores.shape))
# scores.shape: (5000, 100)

print('scores[0]: \n{}'.format(scores[0]))

def get_correlation_coefficents(scores, target_user_index):
    similarities = []
    target = scores[target_user_index]
    
    for i, score in enumerate(scores):
        # 共通の評価が少ない場合は除外
        indices = np.where(((target + 1) * (score + 1)) != 0)[0]
        if len(indices) < 3 or i == target_user_index:
            continue
        
        similarity = np.corrcoef(target[indices], score[indices])[0, 1]
        if np.isnan(similarity):
            continue
    
        similarities.append((i, similarity))
    
    return sorted(similarities, key=lambda s: s[1], reverse=True)


target_user_index = 0 # 0番目のユーザ
similarities = get_correlation_coefficents(scores, target_user_index)

print('Similarities: {}'.format(similarities))
# Similarities: [(186, 1.0), (269, 1.0), (381, 1.0), ...

print('scores[186]:\n{}'.format(scores[186]))


### prediction

def predict(scores, similarities, target_user_index, target_item_index):
    target = scores[target_user_index]
    
    avg_target = np.mean(target[np.where(target >= 0)])
    
    numerator = 0.0
    denominator = 0.0
    k = 0
    
    for similarity in similarities:
        # 類似度の上位5人の評価値を使う
        if k > 5 or similarity[1] <= 0.0:
            break
            
        score = scores[similarity[0]]
        if score[target_item_index] >= 0:
            denominator += similarity[1]
            numerator += similarity[1] * (score[target_item_index] - np.mean(score[np.where(score >= 0)]))
            k += 1
            
    return avg_target + (numerator / denominator) if denominator > 0 else -1


target_item_index = 0 # 3番目のアイテム(エビ)

print('Predict score: {:.3f}'.format(predict(scores, similarities, target_user_index, target_item_index)))
# Predict score: 2.761


def rank_items(scores, similarities, target_user_index):
    rankings = []
    target = scores[target_user_index]
    # 寿司ネタ100種類の全てで評価値を予測
    for i in range(100):
        # 既に評価済みの場合はスキップ
        if target[i] >= 0:
            continue

        rankings.append((i, predict(scores, similarities, target_user_index, i)))
        
    return sorted(rankings, key=lambda r: r[1], reverse=True)


target_user_index = 0 # 0番目のユーザ

rank = rank_items(scores, similarities, target_user_index)
print('Ranking: {}'.format(rank))