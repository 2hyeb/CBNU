import pandas as pd

# 1) 파일 로드
gt = pd.read_excel("ground_truth.xlsx")       # image_name | smoking | drinking | gathering
pred = pd.read_excel("model_output7.xlsx")    # image_name | smoking_pred | drinking_pred | gathering_pred

def extract_num(x):
    # 1) "1.jpg" -> 1
    return int(str(x).split('.')[0])

# 2) image_name 숫자 기준 정렬
pred_sorted = pred.sort_values(
    by="image_name",
    key=lambda s: s.map(extract_num)
).reset_index(drop=True)

pred_sorted.to_excel("model_output_sorted7.xlsx", index=False)

# 3) GT와 예측 병합
df = pd.merge(gt, pred_sorted, on="image_name", how="inner")

# 4) 흡연 정답 여부
df["smoking_score"] = (df["smoking"] == df["smoking_pred"]).astype(int)

# 5) 음주 정답 여부
df["drinking_score"] = (df["drinking"] == df["drinking_pred"]).astype(int)

# 6) 군집 정답 여부
df["gathering_score"] = (df["gathering"] == df["gathering_pred"]).astype(int)

# 7) Accuracy 계산
smoking_accuracy = df["smoking_score"].mean()
print("Smoking accuracy:", smoking_accuracy)

drinking_accuracy = df["drinking_score"].mean()
print("Drinking accuracy:", drinking_accuracy)

gathering_accuracy = df["gathering_score"].mean()
print("Gathering accuracy:", gathering_accuracy)

df.to_excel("eval_result7.xlsx", index=False)