from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model
from tkinter import *
import tkinter as tk
import pandas as pd
import numpy as np
import io

# %% 파일 불러오기

df_x = pd.read_excel('요리 추천기.xlsx', sheet_name='원인')
df_y = pd.read_excel('요리 추천기.xlsx', sheet_name='결과')
# df_recipe = pd.read_excel('요리 추천기.xlsx', sheet_name = '레시피')
print(df_x)
print()
print(df_y)

#%% GoogleColab에서 저장한 모델 불러오기


#모델 불러오기
model = load_model('food_recom_model.h5')

# 데이터 준비
input_data = [ "달걀", "닭고기", "소고기", "돼지고기", "김치", "두부", "당근", 
              "고등어", "생크림", "명란젓", "연어", "떡", "통조림 참치", 
              "양배추", "명태", "불고기", "버섯", "트러플오일", "고구마", 
              "감자", "맛살", "치즈", "새우", "파프리카", "토마토", "닭가슴살", 
              "오이", "바질페스토", "가지"]  # 입력 데이터 정의


#%% tkinter 생성

from tkinter import Tk, Label, Checkbutton, Button, BooleanVar, LEFT, W

def finalize_selection():
    # 리스트 초기화
    global food_list
    food_list = []
    # 각 체크박스의 상태를 확인하고 리스트에 1 또는 0 추가
    for var in vars:
        food_list.append(1 if var.get() else 0)
    # print(food_list)  # 리스트의 내용을 콘솔에 출력
    window.destroy()  # 창 닫기

# 메인 창 설정
window = Tk()
window.title("재료 입력")

# 리스트 초기화
food_list = []

# 체크박스 항목과 상태를 저장할 리스트 및 변수
ingredients = [
    "달걀", "닭고기", "소고기", "돼지고기", "김치", "두부", "당근", "고등어",
    "생크림", "명란젓", "연어", "떡", "통조림 참치", "양배추", "명태",
    "불고기", "버섯", "트러플오일", "고구마", "감자", "맛살", "치즈",
    "새우", "파프리카", "토마토", "닭가슴살", "오이", "바질페스토", "가지"
]



vars = [BooleanVar() for _ in ingredients]

# 레이블 추가
Label(window, text="재료를 선택해주세요", justify=LEFT, padx=20).grid(
    row=0, column=0, columnspan=3, pady=10)

# 체크박스 추가 (3열로 배치)
num_columns = 3
for index, (ingredient, var) in enumerate(zip(ingredients, vars)):
    row = index // num_columns + 1
    column = index % num_columns
    Checkbutton(window, text=ingredient, padx=20, variable=var).grid(
        row=row, column=column, sticky=W)

# "선택완료" 버튼 추가
Button(window, text="선택완료", command=finalize_selection).grid(row=(len(
    ingredients) + num_columns - 1) // num_columns + 1, column=0, columnspan=3, pady=10)


window.mainloop()


# %% 모델에 들어갈 수 있게 데이터 가공

# index 변경
try:
     df_x = df_x.set_index('항목')
     df_y = df_y.set_index('항목')
except:
     pass

# O -> 1 변경
# X -> 0 변경
#df_x = df_x.replace('O', 1).replace('X', 0)
#df_y = df_y.replace('O', 1).replace('X', 0)

# 독립변수명, 종속 변수명 가져오기
varname_x = list(df_x.index)
varname_y = list(df_y.index)
# print('독립변수는 ', varname_x)
# print('종속변수는 ', varname_y)
# print('')

# 독립변수 갯수, 종속변수 갯수, 데이터 갯수 얻어오기
len_x = len(df_x)
len_y = len(df_y)
len_data = len(df_x.columns)
# print('독립변수(출력) 갯수는 ', len_x)
# print('종속변수(출력) 갯수는 ', len_y)
# print('데이터 갯수는 ', len_data)

# 모델에 맞게 출력 모양 변환
x = np.array(df_x).T
y = np.array(df_y).T

x_train = x[:60]
y_train = y[:60]
x_val = x[60:]
y_val = y[60:]
# print('training data 갯수: ')
# print('valdation data 갯수: ')


# %% 결과 예측하기

# 독립변수 순서에 맞게 잘 넣어주세요!
# [ "달걀", "닭고기", "소고기", "돼지고기", "김치", "두부", "당근", "고등어", "생크림", "명란젓", "연어", "떡", "통조림 참치", "양배추", "명태", "불고기", "버섯", "트러플오일", "고구마", "감자", "맛살", "치즈", "새우", "파프리카", "토마토", "닭가슴살", "오이", "바질페스토", "가지"]
y_pred = model.predict(np.array([food_list]))

# 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
# 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1


# 클래스가 35개인 분류기
# 결과 출력
# for i in range(len_y):
#     print(varname_y[i], ' : ', round(y_pred[0][i], 35))

# 유사도 결과를 내림차순으로 출력
# y_pred[0]와 varname_y를 묶어서 (예측 값, 클래스 이름) 쌍으로 만듭니다.
pred_with_names = list(zip(y_pred[0], varname_y))

# 예측 값을 기준으로 내림차순으로 정렬합니다.
sorted_pred_with_names = sorted(pred_with_names, key=lambda x: x[0], reverse=True)

# 정렬된 결과를 출력합니다.
for pred, name in sorted_pred_with_names:
    print(name, ' : ', round(pred, 35))


# %% 유사도 정렬하기

# 유사도 값과 인덱스를 내림차순으로 정렬
sorted_indices = np.argsort(y_pred[0])[::-1]
sorted_values = np.array(y_pred[0])[sorted_indices]

# 가장 높은 유사도 값
highest_value = sorted_values[0]

# 유사도 임계값 (가장 높은 유사도 값의 50% 이상)
threshold = highest_value * 0.5

# 임계값 이상인 인덱스 찾기
filtered_indices = [idx for idx in sorted_indices if y_pred[0][idx] >= threshold]

# 가장 높은 3개의 유사도 값의 인덱스 선택
top_indices = filtered_indices[:3]


# %% 레시피

# 레시피 데이터 딕셔너리
recipe_dict = {
    '명란파스타': '파스타를 삶고 체에 밭쳐놓습니다.\n팬에 올리브유를 두르고 다진 마늘을 볶습니다.\n명란을 넣고 살짝 볶은 후 크림을 추가합니다.\n삶은 파스타를 넣고 잘 섞어줍니다.\n소금과 후추로 간을 하고, 파슬리로 장식합니다.',
    '닭가슴살파스타': '파스타를 삶습니다.\n닭가슴살을 구워서 잘게 썹니다.\n팬에 마늘과 양파를 볶다가 방울토마토를 넣습니다.\n생크림과 치즈를 넣고, 닭가슴살과 삶은 파스타를 넣어 섞습니다.',
    '불고기파스타': '파스타를 삶습니다.\n불고기용 소고기를 간장, 설탕, 마늘로 재운 후 볶습니다.\n양파와 파프리카를 추가하여 볶습니다.\n삶은 파스타를 넣고 섞습니다.',
    '크림파스타': '파스타를 삶습니다.\n팬에 마늘과 양파를 볶다가 버섯을 추가합니다.\n생크림을 넣고 끓입니다.\n삶은 파스타를 넣고 치즈를 뿌려 섞습니다.',
    '바질파스타': '파스타를 삶습니다.\n팬에 올리브유와 마늘을 볶습니다.\n삶은 파스타를 넣고 바질과 치즈를 추가하여 섞습니다.',
    '궁중떡볶이': '떡을 끓는 물에 데칩니다.\n팬에 야채를 볶다가 오뎅을 추가합니다.\n고추장, 간장, 설탕으로 양념하고 떡을 넣어 볶습니다.',
    '바질떡볶이': '떡을 끓는 물에 데칩니다.\n팬에 올리브유와 양파를 볶습니다.\n고추장으로 양념하고 바질을 추가합니다.\n떡을 넣고 잘 섞어줍니다.',
    '로제떡볶이': '떡을 데칩니다.\n팬에 양파를 볶다가 로제 소스를 추가합니다.\n떡을 넣고 잘 섞어줍니다.',
    '토마토수프': '양파와 마늘을 볶다가 다진 토마토를 추가합니다.\n물을 넣고 끓입니다.\n소금과 후추로 간을 맞추고, 믹서기로 갈아줍니다.',
    '게살스프': '양파와 마늘을 볶다가 게살을 추가합니다.\n치킨 스톡과 생크림을 넣고 끓입니다.\n소금과 후추로 간을 맞추고, 믹서기로 갈아줍니다.',
    '감자치즈스튜': '감자와 당근을 잘라서 볶습니다.\n양파를 추가하고 육수를 붓습니다.\n끓이다가 치즈를 넣고 녹입니다.',
    '토마토계란볶음밥': '양파와 토마토를 볶다가 계란을 추가하여 스크램블합니다.\n밥을 넣고 잘 섞어줍니다.\n소금과 후추로 간을 맞춥니다.',
    '새우볶음밥': '새우와 채소를 볶다가 밥을 추가합니다.\n간장으로 간을 맞추고 잘 섞어줍니다.',
    '불고기볶음밥': '불고기와 채소를 볶다가 밥을 추가합니다.\n간장으로 간을 맞추고 잘 섞어줍니다.',
    '두부조림': '두부를 구워서 적당한 크기로 자릅니다.\n팬에 간장, 설탕, 마늘을 섞어 끓입니다.\n두부를 넣고 조립니다.\n참기름과 파를 넣어 마무리합니다.',
    '고등어조림': '고등어를 양념(간장, 고춧가루, 마늘, 생강)하여 팬에 조립니다.\n양파를 넣고 함께 조립니다.',
    '오코노미야키': '밀가루, 계란, 물을 섞어 반죽을 만듭니다.\n양배추와 베이컨을 섞어 반죽에 추가합니다.\n팬에 부쳐서 양면이 노릇하게 구워줍니다.\n오코노미야키 소스와 마요네즈를 뿌려 제공합니다.',
    '연어초밥': '초밥 밥을 작은 덩어리로 만들어 놓습니다.\n연어를 얇게 썰어 밥 위에 올립니다.\n와사비와 간장과 함께 제공합니다.',
    '연어덮밥': '연어를 간장, 미림, 설탕에 재운 후 팬에 구워줍니다.\n밥 위에 구운 연어를 올리고, 파와 참기름을 뿌립니다.',
    '참치마요덮밥': '참치를 마요네즈와 섞습니다.\n밥 위에 참치 마요를 올리고, 간장과 양파, 파를 뿌립니다.',
    '연어솥밥': '연어를 간장과 미림에 재운 후 팬에 구워줍니다.\n쌀을 씻어 물과 함께 솥에 넣고 끓입니다.\n구운 연어와 버섯을 추가하고 잘 섞어줍니다.\n파를 뿌려 마무리합니다.',
    '버섯솥밥': '쌀을 씻어 물과 함께 솥에 넣습니다.\n버섯과 다진 마늘을 추가하고 간장으로 간을 맞춥니다.\n끓인 후 파를 뿌려 마무리합니다.',
    '김치부침개': '김치를 잘게 썰고 밀가루와 물, 계란을 섞어 반죽을 만듭니다.\n팬에 반죽을 부어 노릇하게 구워줍니다.\n대파를 뿌려 제공합니다.',
    '육전': '소고기를 얇게 썰어 소금과 후추로 간합니다.\n밀가루와 계란을 묻혀 팬에 구워줍니다.',
    '명태국': '명태를 적당한 크기로 자릅니다.\n육수에 명태, 대파, 마늘, 고춧가루를 넣고 끓입니다.\n간장으로 간을 맞추고 제공합니다.',
    '소고기국': '소고기를 적당한 크기로 자릅니다.\n육수에 소고기, 대파, 마늘을 넣고 끓입니다.\n간장과 후추로 간을 맞추고 제공합니다.',
    '김치찌개': '김치와 돼지고기를 볶다가 물을 추가합니다.\n끓으면 두부와 대파, 마늘, 고춧가루를 넣습니다.\n간장으로 간을 맞추고 제공합니다.',
    '가지돼지고기볶음': '가지와 돼지고기를 적당한 크기로 자릅니다.\n팬에 돼지고기를 볶다가 가지를 추가합니다.\n간장, 설탕, 마늘로 양념하고 볶습니다.\n대파를 넣어 마무리합니다.',
    '가지볶음': '가지를 적당한 크기로 자릅니다.\n팬에 올리브유와 마늘을 볶다가 가지를 넣습니다.\n간장과 설탕으로 양념하고 볶습니다.',
    '오이무침': '오이를 얇게 썰어 소금에 절입니다.\n물기를 제거하고, 고춧가루, 식초, 설탕, 다진 마늘로 양념합니다.',
    '스테이크': '고기에 소금과 후추로 간을 합니다.\n팬에 버터와 마늘을 넣고 고기를 구워줍니다.',
    '오므라이스': '팬에 양파와 햄을 볶다가 밥을 추가합니다.\n케찹으로 간을 맞추고 섞습니다.\n계란을 풀어 팬에 부쳐 밥 위에 덮습니다.',
    '계란말이': '계란에 소금과 후추를 넣고 잘 풀어줍니다.\n팬에 식용유를 두르고 계란을 얇게 부쳐서 말아줍니다.\n대파를 넣어 함께 말아줍니다.',
    '고구마맛탕': '고구마를 적당한 크기로 자릅니다.\n팬에 식용유를 두르고 고구마를 튀깁니다.\n설탕과 물엿을 섞어 고구마에 버무립니다.',
    '돈가스': '돼지고기에 소금과 후추로 간을 합니다.\n밀가루, 계란, 빵가루를 순서대로 묻혀 팬에 튀깁니다.'
}

# %% 결과 출력

# if top_indices:
#     for rank, idx in enumerate(top_indices, start=1):
#         dish_name = varname_y[idx]
#         recipe = recipe_dict.get(dish_name, '레시피를 찾을 수 없습니다.')
#         print(f"{rank}. {dish_name}: {recipe}")
# else:
#     print("죄송합니다. 올바른 결과를 찾지 못하였습니다.")


# %% tkinter로 결과 
import requests
import os
import openai

# OpenAI API 설정
openai.api_key = ""

# Tkinter 윈도우 생성
root = tk.Tk()
root.title("음식 레시피")

# 제목 레이블 추가
title_font = ("Helvetica", 16, "bold")
title_label = tk.Label(root, text="추천하는 음식과 레시피", font=title_font, pady=10)
title_label.pack()

# 텍스트 위젯 생성
text_widget = tk.Text(root, wrap='word', height=20, width=50, font=("Arial", 12), bg="#f0f0f0", fg="#333", padx=10, pady=10)
text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# 스크롤바 추가
scrollbar = tk.Scrollbar(root, command=text_widget.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# 텍스트 위젯과 스크롤바 연결
text_widget.config(yscrollcommand=scrollbar.set)

# 리스트의 값을 문자열로 변환
output_list = [input_data[i] for i, value in enumerate(food_list) if value == 1]



# ChatGPT 레시피 추천 함수
def recommend_recipe():
    prompt = (
        "다음 재료를 사용하여 창의적이고 맛있는 요리 레시피를 추천해 주세요. "
        f"재료: {output_list}"
    )
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "너는 요리사야"},
                {"role": "user", "content": prompt},
            ]
        )
        recipe = response['choices'][0]['message']['content'].strip()
        text_widget.insert(tk.END, f"\n\n-----------\n\nChatGPT의 추천 레시피:\n{recipe}\n\n")
    except Exception as e:
        text_widget.insert(tk.END, f"\n\n-----------\n\n예외 발생: {str(e)}\n\n")


    
# 결과 출력
def display_results():
    if top_indices:
        output = f"입력된 재료: {output_list}\n\n"
        for rank, idx in enumerate(top_indices, start=1):
            dish_name = varname_y[idx]
            recipe = recipe_dict.get(dish_name, '레시피를 찾을 수 없습니다.')
            output += f"{rank}. {dish_name}: {recipe}\n\n"  # 항목 간에 여백 추가
        text_widget.insert(tk.END, output)
        
        
    else:
        text_widget.insert(tk.END, "죄송합니다. 올바른 결과를 찾지 못하였습니다.")

display_results()


# 버튼 스타일 설정
recommend_button = tk.Button(
    root,
    text="ChatGPT 레시피 추천",
    command=recommend_recipe,
    font=("Arial", 12, "bold"),
    bg="#003366",  # 네이비 블루
    fg="#87CEEB",  # 스카이 블루
    relief="raised",
    padx=20,
    pady=10,
    borderwidth=2,
    highlightbackground="#003366",
    highlightthickness=2
)

# 버튼 배치
recommend_button.pack(pady=20)

# Tkinter 이벤트 루프 시작
root.mainloop()


# %% chatGPT 활용

# import os
# import openai

# openai.api_key = 

# response = openai.ChatCompletion.create(
#   model="gpt-4",
#   messages=[
#         {"role": "system", "content": "너는 요리사야"},
#         {"role": "user", "content": "닭가슴살만으로 해물파스타 레시피 만들어줘"},
#     ]
# )

# print(response['choices'][0]['message']['content'])
