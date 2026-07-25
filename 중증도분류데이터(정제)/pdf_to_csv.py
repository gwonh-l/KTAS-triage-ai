import tabula

# input.pdf 파일을 읽어서 output.csv 파일로 변환
# pages='all'을 설정하면 PDF 전체 페이지의 표를 다 가져옴 (특정 페이지만 원하면 pages='1' 처럼 숫자 입력)
tabula.convert_into("kid.pdf", "pediatric_raw.csv", output_format="csv", pages='all')

print("CSV 변환 완료!")