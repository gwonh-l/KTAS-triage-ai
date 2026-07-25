import pandas as pd

# 1. 저장한 CSV 파일 불러오기
# (한글이 깨질 경우 encoding='utf-8' 대신 'cp949'나 'utf-8-sig'로 변경)
df_adult = pd.read_csv('adult_raw.csv')
df_pediatric = pd.read_csv('pediatric_raw.csv')

# 2. 연령대(age_group) 컬럼을 맨 앞에 추가
df_adult.insert(0, 'age_group', '15세 이상')
df_pediatric.insert(0, 'age_group', '15세 미만')

# 3. 두 데이터프레임 병합 (위아래로 합치기)
df_final = pd.concat([df_adult, df_pediatric], ignore_index=True)

# 4. 빈 줄이나 이상한 데이터 정제 (안전 장치)
df_final = df_final.dropna(subset=['symptom', 'ktas_level'])
df_final['ktas_level'] = pd.to_numeric(df_final['ktas_level'], errors='coerce').fillna(0).astype(int)
df_final = df_final[(df_final['ktas_level'] >= 1) & (df_final['ktas_level'] <= 5)]

# 5. 최종 결과물을 새로운 CSV 파일로 저장
output_filename = 'ktas_clean_dataset.csv'
df_final.to_csv(output_filename, index=False, encoding='utf-8-sig')

print(f"가공 완료! 총 {len(df_final)}개의 데이터가 '{output_filename}'로 저장되었습니다.")
print("\n[최종 데이터 미리보기]")
print(df_final.head())