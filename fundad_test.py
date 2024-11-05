import matplotlib
matplotlib.use('Agg')
import numpy as np
import os
import pandas as pd
from pymongo import MongoClient
import pymongo
import re
import riskfolio as rp
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import io
from matplotlib import rc
import json
import base64
from io import BytesIO
import certifi
from dotenv import load_dotenv
import cvxpy as cp
from db import dbclient ,client
import unicodedata
import matplotlib.font_manager as fm
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta
import seaborn as sns
import numpy as np
import plotly.io as pio  # Plotly의 IO 모듈을 가져옵니다.
import matplotlib.ticker as mticker
from datetime import datetime
from dateutil.relativedelta import relativedelta




# 폰트 설정
matplotlib.rcParams['font.family'] = 'Malgun Gothic'

#################################### 펀드 데이터 처리 ####################################                  
db = dbclient['fund_data']  # 데이터베이스 이름
collection = db['fund_info']
# 펀드 데이터 불러오기                                              
data_f = list(collection.find())
dff = pd.DataFrame(data_f)
# '펀드 정보' 열에서 6개월 이상 수익률 데이터가 있는 펀드만 남김
dff['수익률_개월수'] = dff['펀드 정보'].apply(lambda x: len([entry for entry in x if '수익률' in entry]))
dff = dff[dff['수익률_개월수'] >= 6].drop(columns='수익률_개월수')


#################################### 섹터 처리 ####################################
def initialize_base_sectors(dff):
    # 리스트 자체의 중복 제거, 빈 리스트 제거 후 주요 섹터 리스트 형태 유지
    unique_sectors = []
    seen = set()

    for sublist in dff['주요 섹터']:
        if isinstance(sublist, list):
            cleaned_sublist = tuple(sector.strip() for sector in sublist if isinstance(sector, str) and sector.strip() != 'None' and sector not in ['레버리지', '인버스'])
            if cleaned_sublist and cleaned_sublist not in seen:
                unique_sectors.append(list(cleaned_sublist))
                seen.add(cleaned_sublist)
    unique_sectors.append(['TDF'])
    return unique_sectors
base_sectors = initialize_base_sectors(dff)

# 등급에 따라 섹터 추가
def get_unique_sectors_by_grade(grade):
    """
    등급에 따라 unique_sectors를 생성하는 함수
    """
    unique_sectors = base_sectors.copy()

    # 1등급(공격투자형)일 때만 레버리지와 인버스를 각각 독립적으로 추가
    if grade == 1:
        unique_sectors.append(['레버리지'])
        unique_sectors.append(['인버스'])
    return unique_sectors

# 등급에 따른 평균 수익률과 표준편차 데이터 생성
grade_data = pd.DataFrame({
    '등급': [1, 2, 3, 4, 5, 6],
    '평균 수익률': [0.058317, 0.094167, 0.079773, 0.063759, 0.032135, 0.025660],
    '표준편차': [0.128992, 0.072929, 0.059175, 0.033801, 0.015387, 0.001350]
})

#################################### 섹터 분류 #################################### 
def classify_sector(user_input, user_profile):
    user_age = user_profile['age']
    user_grade = user_profile['grade']
    unique_sectors = get_unique_sectors_by_grade(user_grade)
    response = client.chat.completions.create(
    model="gpt-4-turbo",
    messages=[
            {"role": "system", "content": f"넌 펀드 추천 질문에서 섹터를 찾아서 분류할 예정이야. 섹터는 {unique_sectors} 이거야.\
            '채권'만 입력되었다면 '채권'으로 답변해줘.\
            'TDF' 또는 'tdf'만 입력되었다면 'TDF'으로 답변해줘\
            '인덱스'만 입력되었을 때는 0을 반환해, '인덱스-일본'과 같은 형태로 입력되었을 때는 '인덱스', '일본'을 반환해줘."},
            {"role": "user", "content": f"이 질문을 펀드 섹터에 맞게 분류해줘.: '{user_input}'"}
    ],
    max_tokens=50,
    temperature=0
    )
    classification = response.choices[0].message.content.strip()
    # TDF 단독 입력 시 나이에 맞는 TDF 추천 반환
    if classification == 'TDF':
        return assign_tdf_by_age(user_age)
    elif classification == '채권':
        return '채권'
    # TDF 단독이 아닌 경우 GPT 분류 결과를 그대로 리스트로 반환
    sector_list = [[sector.strip() for sector in classification.split(',')]]
    return sector_list if sector_list else 0

#################################### 나이에 따른 TDF 추천 ####################################
def assign_tdf_by_age(age):
    """
    나이에 따른 TDF 추천
    """
    if age < 30:
        return [['TDF', '2050'], ['TDF', '2055'], ['TDF', '2060']]
    elif 30 <= age < 40:
        return [['TDF', '2040'], ['TDF', '2045'], ['TDF', '2050']]
    elif 40 <= age < 50:
        return [['TDF', '2030'], ['TDF', '2035'], ['TDF', '2040']]
    elif 50 <= age < 60:
        return [['TDF', '2020'], ['TDF', '2025'], ['TDF', '2030']]
    else:
        return [['TDF', '2015'], ['TDF', '2020'], ['TDF', '2025']]
    
#################################### 평균 수익률 계산 ####################################
def calculate_average_return(fund_df):
    """
    각 펀드의 1년 수익률 계산 (최신 2년 데이터만 사용, 24개월 미만의 데이터가 있는 경우 None 반환)
    """
    # 데이터가 24개월 미만이면 None 반환
    if len(fund_df) < 24:
        return None

    # 기준연월을 날짜 형식으로 변환하여 정렬 후 최신 2년 데이터만 추출
    fund_df['기준연월'] = pd.to_datetime(fund_df['기준연월'], format='%Y-%m')
    fund_df = fund_df.sort_values('기준연월', ascending=False).head(24)

    # 1년 전 기준가격을 기준으로 1년 수익률 계산
    fund_df['1년 전 기준가격'] = fund_df['기준가격'].shift(-12)
    fund_df['1년 수익률'] = (fund_df['기준가격'] - fund_df['1년 전 기준가격']) / fund_df['1년 전 기준가격']

    # 최근 12개월의 1년 수익률 평균 계산 (NaN 제외)
    recent_12_months = fund_df.head(12)
    return recent_12_months['1년 수익률'].mean()

#################################### 표준편차 계산 ####################################
def calculate_std_dev(fund_df):
    """
    각 펀드의 1년 수익률 표준편차 계산 (최신 2년 데이터만 사용, 24개월 미만의 데이터가 있는 경우 None 반환)
    """
    # 데이터가 24개월 미만이면 None 반환
    if len(fund_df) < 24:
        return None

    # 기준연월을 날짜 형식으로 변환하여 정렬 후 최신 2년 데이터만 추출
    fund_df['기준연월'] = pd.to_datetime(fund_df['기준연월'], format='%Y-%m')
    fund_df = fund_df.sort_values('기준연월', ascending=False).head(24)

    # 1년 전 기준가격을 기준으로 1년 수익률 계산
    fund_df['1년 전 기준가격'] = fund_df['기준가격'].shift(-12)
    fund_df['1년 수익률'] = (fund_df['기준가격'] - fund_df['1년 전 기준가격']) / fund_df['1년 전 기준가격']

    # 최근 12개월의 1년 수익률 표준편차 계산 (NaN 제외)
    recent_12_months = fund_df.head(12)
    return recent_12_months['1년 수익률'].dropna().std()


#################################### 기울기 계산 ####################################
def calculate_slope(df):
    return df['평균 수익률'] / df['표준편차']
grade_data['기울기'] = calculate_slope(grade_data)

# 펀드별 기울기 계산 함수
def calculate_slope_fund(avg_return, std_dev):
    if std_dev == 0:  # 분모가 0이 되는 것을 방지
        return None
    return avg_return / std_dev

#################################### 펀드 추천 ####################################
def recommend_fund(user_profile, sector_list):
    """
    사용자 프로필과 섹터 정보를 이용하여 펀드를 추천하는 함수
    """
    # 사용자 등급에 맞는 펀드 필터링
    user_grade = user_profile['grade']
    # dff에서 주요 섹터가 sector_list의 어떤 리스트와 일치하는 펀드 필터링
    sector_funds = dff[dff['주요 섹터'].apply(
        lambda x: any(x == s for s in sector_list) if isinstance(x, list) else False
    )].copy()

    # 만약 필터링된 섹터 펀드가 없을 경우 None 반환
    if sector_funds.empty:
        return None

    sector_funds.loc[:, '평균 수익률'] = sector_funds['펀드 정보'].apply(lambda x: calculate_average_return(pd.DataFrame(x)))
    sector_funds.loc[:, '표준편차'] = sector_funds['펀드 정보'].apply(lambda x: calculate_std_dev(pd.DataFrame(x)))
    # 평균 수익률과 표준편차가 None이 아닌 펀드만 필터링
    sector_funds = sector_funds[(sector_funds['평균 수익률'].notnull()) & (sector_funds['표준편차'].notnull())]

    # 기울기 계산
    sector_funds.loc[:, '기울기'] = sector_funds.apply(lambda row: calculate_slope_fund(row['평균 수익률'], row['표준편차']), axis=1)

    # 사용자 기울기 계산
    grade_slopes = grade_data.set_index('등급')['기울기']
    if user_grade == 1:
        user_slope = (grade_slopes.get(1) + grade_slopes.get(2)) / 2
    elif user_grade in [2, 3, 4, 5]:
        user_slope = grade_slopes.get(user_grade, None)
    else:
        return None

    higher_slope_funds = sector_funds[sector_funds['기울기'] > user_slope]

    if higher_slope_funds.empty:
        sector_funds.loc[:, '기울기 차이'] = abs(sector_funds['기울기'] - user_slope)
        if sector_funds.empty:
            return None
        closest_fund_name = sector_funds.sort_values('기울기 차이').iloc[0]['펀드명']
        return closest_fund_name
    else:
        if user_grade in [3, 4, 5]:
            top_fund_name = higher_slope_funds.sort_values('기울기', ascending=False).iloc[0]['펀드명']
        else:
            max_slope_fund = higher_slope_funds.sort_values('기울기', ascending=False).iloc[0]
            max_slope_value = max_slope_fund['기울기']
            lower_slope_threshold = max_slope_value - sector_funds.loc[:, '기울기'].std() * 1
            low_volatility_funds = sector_funds[
                (sector_funds['기울기'] > lower_slope_threshold) &
                (sector_funds['기울기'] > user_slope)
            ]

            if not low_volatility_funds.empty:
                top_fund_name = low_volatility_funds.sort_values('평균 수익률', ascending=False).iloc[0]['펀드명']
            else:
                top_fund_name = higher_slope_funds.sort_values('기울기').iloc[0]['펀드명']

    return top_fund_name

#################################### 상관계수가 낮은 펀드 찾기 ####################################
def find_top_negative_correlations(df, target_fund_name,user_profile):
    user_grade = user_profile['grade']
    # 정해진 등급 내 펀드 필터링
    filtered_df = df.copy()

    # 타겟 펀드의 수익률 데이터 추출
    target_fund_data = filtered_df[filtered_df['펀드명'] == target_fund_name]

    # 타겟 펀드가 없는 경우 처리
    if target_fund_data.empty:
        return None

    target_fund_returns = target_fund_data.iloc[0]['펀드 정보']
    target_fund_returns_df = pd.DataFrame(target_fund_returns)

    # 사용자의 등급에 따라 적절한 섹터 목록 가져오기
    unique_sectors = get_unique_sectors_by_grade(user_grade)
    filtered_df = filtered_df[filtered_df['주요 섹터'].isin(unique_sectors)]

    # 상관계수 계산
    correlations = {}
    for _, row in filtered_df.iterrows():
        fund_name = row['펀드명']
        fund_returns = pd.DataFrame(row['펀드 정보'])['수익률']
        aligned_target, aligned_fund = target_fund_returns_df['수익률'].align(fund_returns, join='inner')
        correlation = aligned_target.corr(aligned_fund)
        correlations[fund_name] = correlation

    # 상관계수가 음수인 것만 필터링하여 정렬
    negative_correlations_df = pd.DataFrame(list(correlations.items()), columns=['펀드명', '상관계수'])
    negative_correlations_df = negative_correlations_df[negative_correlations_df['상관계수'] < 0].sort_values(by='상관계수')

    # 상위 3개의 상관계수가 낮은 펀드 선택
    top_negative_correlations = negative_correlations_df.head(3)

    return top_negative_correlations

#################################### 포트폴리오 최적화 ####################################
def riskfolio(topfund, corfunds):
    # topfund와 corfunds에 있는 펀드명 추출
    funds = [topfund] + corfunds['펀드명'].tolist()

    # dff에서 해당 펀드명들만 필터링
    filtered_funds = dff[dff['펀드명'].isin(funds)]

    # 각 펀드의 전체 데이터를 사용하여 데이터 프레임 생성
    recent_data = []
    for _, row in filtered_funds.iterrows():
        fund_info = row['펀드 정보']  # 리스트 형태의 펀드 정보 추출
        if isinstance(fund_info, list):
            for data in fund_info:
                recent_data.append({'펀드명': row['펀드명'], '기준연월': data['기준연월'], '수익률': data.get('수익률', 0)})

    # 데이터프레임 생성
    recent_6_months = pd.DataFrame(recent_data)

    # '수익률' 컬럼이 있는지 확인하고, 없으면 예외 처리
    if '수익률' not in recent_6_months.columns or recent_6_months.empty:
        raise ValueError("수익률 데이터가 충분하지 않거나 유효하지 않습니다. 필터링 후 남아있는 데이터가 없습니다.")

    # 기준연월을 datetime 형식으로 변환하여 정렬
    recent_6_months['기준연월'] = pd.to_datetime(recent_6_months['기준연월'], format='%Y-%m')
    recent_6_months = recent_6_months.sort_values(by='기준연월')

    # 최신 데이터 제외한 최근 6개월 데이터만 사용하도록 필터링
    latest_date = recent_6_months['기준연월'].max()
    recent_6_months = recent_6_months[recent_6_months['기준연월'] < latest_date]
    recent_6_months = recent_6_months.sort_values(by='기준연월', ascending=False).groupby('펀드명').head(6)

    # 기준연월을 오름차순으로 정렬
    recent_6_months = recent_6_months.sort_values(by='기준연월')

    # 기준연월을 인덱스로, 펀드명을 열로, 수익률을 값으로 피벗 테이블 생성
    fund_returns_6_months = recent_6_months.pivot_table(index='기준연월', columns='펀드명', values='수익률', aggfunc='mean')

    if fund_returns_6_months.empty:
        raise ValueError("수익률 데이터가 충분하지 않습니다. 필터링 후 남아있는 데이터가 없습니다.")

    # 분산이 0인 경우 필터링
    fund_returns_6_months = fund_returns_6_months.loc[:, fund_returns_6_months.std() > 0]

    # Riskfolio-Lib을 사용하여 최적화
    port = rp.Portfolio(returns=fund_returns_6_months)

    # 자산 수익률 및 공분산 계산 (평균 수익률 및 공분산 계산)
    port.assets_stats(method_mu='hist', method_cov='ewma1')

    # 각 펀드의 최소 및 최대 가중치 설정 (topfund는 최소 10% 보장)
    w_min = [0.1 if fund == topfund else 0.01 for fund in funds]
    w_max = [0.5] * len(funds)
    port.w_min = np.array(w_min)
    port.w_max = np.array(w_max)

    # 가중치 범위 설정
    port.bounds = (0.05, 0.5)

    # 최적화 수행 (샤프 비율 최대화)
    w = port.optimization(model='Classic', rm='MV', obj='MinRisk', hist=True)

    # 최적화 결과가 유효한지 확인
    if w is None or 'weights' not in w:
        raise ValueError("최적화 결과가 유효하지 않습니다. 포트폴리오 가중치를 계산할 수 없습니다.")

    # 1% 이하 가중치 제거
    w = w[w['weights'] > 0.01]

    if topfund not in w.index:
        new_row = pd.DataFrame({'weights': [0.1]}, index=[topfund])
        w = pd.concat([w, new_row])

        # 가중치의 합 계산
        total_weight = w['weights'].sum()
        # 각 가중치를 합으로 나누어 합이 1이 되도록 스케일링
        w['weights'] = w['weights'] / total_weight

    # plt.pie()로 차트 생성
    plt.figure(figsize=(35, 20))  # 차트 크기 설정
    plt.pie(w['weights'], labels=w.index, autopct='%1.1f%%', startangle=90, labeldistance=1.05, 
            textprops={'fontsize': 30})
    plt.rc('font', size=20)
  
    # 제목 추가
    plt.title('Optimized Portfolio Weights')

    # 이미지 저장 및 인코딩
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    image_data = base64.b64encode(buf.getvalue()).decode('utf-8')

    # 최적화된 가중치 반환
    return w, {'image': image_data}  # Base64 인코딩된 이미지 데이터

#################################### 펀드 정보 추출 ####################################
def extract_fund_name(user_input):
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "사용자가 입력한 질문에서 펀드명을 추출해줘. 펀드명은 질문에서 언급된 고유명사로, 펀드와 관련된 중요한 키워드야.다른 설명 없이 펀드명만 딱 대답해줘"},
            {"role": "user", "content": f"질문: '{user_input}'"}
        ],
        max_tokens=50,
        temperature=0
    )
    fund_name = response.choices[0].message.content.strip()
    return fund_name

#################################### 펀드 정보 추출 ####################################
def get_fund_info(user_input):
    """
    사용자 입력을 기준으로 전역 변수 dff에서 매칭되는 펀드를 검색하여 정보를 반환하는 함수
    """
    # 사용자 입력을 소문자 및 정규화 처리
    user_input = user_input.strip().lower().replace(" ", "")

    # dff에서 '펀드명' 열이 user_input을 포함하는 항목을 검색
    matching_funds = dff[dff['펀드명'].str.lower().str.replace(" ", "").str.contains(user_input, na=False, regex=False)]

    # 매칭되는 펀드가 있으면 첫 번째 결과의 정보를 반환, 없으면 None 반환
    if not matching_funds.empty:
        fund_info = matching_funds.iloc[0].to_dict()  # 첫 번째 매칭 항목을 딕셔너리로 반환
        return fund_info
    else:
        return None
    
#################################### 질문 기간 추출 ####################################
def extract_monthly_dates_from_question(question):
    """
    사용자 질문을 기반으로 월 단위의 startdate와 enddate를 YYYY-MM 형식의 문자열로 반환.
    질문에 기간 정보가 없을 경우 최신 데이터인 9월을 기준으로 최근 12개월 데이터를 반환.
    """
    # 최신 데이터가 9월임을 GPT에 명시
    latest_data_month = "2024-09"

    # GPT 프롬프트 생성
    prompt = f"""
    사용자가 요청한 기간에 따라 월 단위의 startdate와 enddate를 YYYY-MM 형식으로 반환하세요.
    - 최신 데이터는 {latest_data_month}임을 참고하여 이 범위 내에서 기간을 계산합니다.
    - 예를 들어 "최근 1년"을 요청했다면 2023-09부터 2024-09까지를 반환합니다.
    - 질문에 기간 정보가 없을 경우 최근 12개월 데이터를 반환하세요.
    -"YYYY-MM","YYYY-MM"으로만 반환하세요
    질문: "{question}"
    """

    # GPT API 호출
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "당신은 월 단위의 기간을 추출하는 전문가입니다."},
            {"role": "user", "content": prompt}
        ],
            max_tokens=50,
            temperature=0
    )

    # GPT로부터 받은 응답을 문자열로 처리
    date_info = response.choices[0].message.content.strip()

    # GPT 응답이 비어 있거나 형식이 잘못된 경우 기본 기간으로 설정
    if not date_info or ',' not in date_info:
        enddate = latest_data_month  # 최신 데이터인 9월을 기준으로
        startdate = (datetime.strptime(latest_data_month, '%Y-%m') - relativedelta(months=12)).strftime('%Y-%m')  # 6개월 전
    else:
        # 날짜 형식이 올바른지 확인하고 분할
        dates = date_info.split(',')
        startdate = dates[0].strip()
        enddate = dates[1].strip()

    return startdate, enddate

#################################### 펀드 기준가격 그래프 생성 ####################################
def generate_price_graph(fund_info, startdate, enddate):
    """
    특정 기간(startdate~enddate)에 맞춘 펀드 기준가격 꺾은선 그래프 생성 함수
    """
    # '펀드 정보'에서 기준가격 데이터를 데이터프레임으로 변환
  # 데이터 준비
    data = pd.DataFrame(fund_info.get("펀드 정보", []))
    data['기준연월'] = pd.to_datetime(data['기준연월'], format='%Y-%m')
    startdate_dt = pd.to_datetime(startdate.strip().replace('"', ''), format='%Y-%m')
    enddate_dt = pd.to_datetime(enddate.strip().replace('"', ''), format='%Y-%m')

    filtered_data = data[(data['기준연월'] >= startdate_dt) & (data['기준연월'] <= enddate_dt)]
    dates = filtered_data['기준연월'].dt.strftime('%Y-%m').tolist()
    prices = filtered_data['기준가격'].tolist()
    scales = (filtered_data['설정원본'] * filtered_data['기준가격']).tolist()  # 규모 계산 (설정원본 * 기준가격)
    settings = (filtered_data['설정원본'] / 1e8).tolist()  # 설정액을 억 단위로 변환

    # 그래프 생성
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 기준가격 축 설정 (빨강색 라인)
    ax1.plot(dates, prices, marker='o', color='#ff4c4c', linestyle='-', linewidth=2, markersize=6, label="기준가격")
    ax1.set_xlabel("기준연월", fontsize=14, labelpad=10, color='#333333')
    ax1.set_ylabel("기준가격", fontsize=14, color='#ff4c4c')
    ax1.tick_params(axis='y', labelcolor='#ff4c4c')
    ax1.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

    # 설정액 축 설정 (파랑색 라인)
    ax2 = ax1.twinx()
    ax2.plot(dates, settings, marker='s', color='#4c72ff', linestyle='--', linewidth=2, markersize=6, label="설정액(억 원)")
    ax2.set_ylabel("설정액 (억 원)", fontsize=14, color='#4c72ff')
    ax2.tick_params(axis='y', labelcolor='#4c72ff')

    # 설정액 축 포맷팅
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))  # 억 단위로 표시

    # 제목과 스타일 설정
    plt.title(f"{startdate} ~ {enddate} 펀드 기준가격 및 규모 추이", fontsize=16, fontweight='bold', color='#333333')
    plt.xticks(fontsize=10, rotation=45, ha='right', color='#333333')

    # 불필요한 테두리 제거
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    # 이미지 저장 및 인코딩
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    image_data = base64.b64encode(buf.getvalue()).decode('utf-8')

    return {'image': image_data}

#################################### 질문 분류 #################################### 
def classify_question(user_input):
    response = client.chat.completions.create(
    model="gpt-4-turbo",
    messages=[
            {
                "role": "system",
                "content": (
                    "넌 질문을 정해주는 카테고리 별로 분류해서 정해진 답을 출력할거야. "
                    "질문은 다음 세 가지 카테고리 중 하나로 분류해야 해:\n"
                    "1. 섹터별 펀드 추천 (R): '추천'이라는 단어가 들어가거나, 특정 섹터를 기준으로 펀드 추천을 요청하는 경우.\n"
                    "2. 펀드 명을 통한 펀드 정보 출력 (F): 질문에 특정 펀드명이 포함된 경우. 펀드 명은 고유명사야"
                    "예시: '미래에셋퇴직연금미국리츠40증권자투자신탁에 대한 펀드 정보 알려줄래'.\n"
                    "3. 그 외 경제 관련 질문이 아닌 경우 (N).\n"
                    "펀드 명을 포함한 질문은 항상 F로 분류하고, '추천'이라는 단어가 들어가면 항상 R로 분류해줘."
                )
            },
            {"role": "user", "content": f"이 질문을 분류해줘: '{user_input}'"}
        ],
    max_tokens=10,
    temperature=0
    )
    classification = response.choices[0].message.content.strip().upper()
    return classification

#################################### 표준 챗봇 ####################################
def standard_chatbot(user_input):
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "사용자의 질문이 들어오면 gpt의 기능을 활용하여 잘 대답해주지만 펀드에 관한 질문을 해달라고 유도해. 펀드나 투자 관련 질문(용어 설명 등)이라면 대답해줘"},
            {"role": "user", "content": f"'{user_input}'"}
        ],
        max_tokens=500,
        temperature=0
    )
    standard = response.choices[0].message.content.strip().upper()
    return standard

#################################### 최종 추천 메시지 생성 ####################################
def generate_final_response(user_grade, sector, top_fund, weight):
    response = client.chat.completions.create(
    model="gpt-4-turbo",
    messages=[
        {"role": "system", "content": "당신은 펀드 추천을 전문으로 하는 금융 어시스턴트입니다. 사용자 정보(위험 등급, 섹터 선호도, 추천 펀드 등)가 주어지면 사용자에게 답변을 내면 됩니다. 다만 너무 딱딱하지만 않게 말해주시고 상관계수를 일일이 출력하지 않아도 됩니다."},
        {"role": "user", "content": f"사용자의 투자성향 등급은 {user_grade}입니다. 1등급이 가장 공격적인 투자성향을 의미하며 가장 안정적인 투자를 원하는 성향은 5등급입니다."
        f"사용자는 {sector} 섹터를 선택했습니다. "
        f"사용자 성향과 가장 맞는 펀드는 {top_fund}입니다. "
        f"또한 {top_fund}와 음의 상관관계를 가진 다른 펀드들의 가중치 비율이 기록된 {weight}가 있습니다. "
        f"이 정보를 바탕으로 사용자에게 최종 추천 메시지를 간결하게 생성해주세요."
        f"가중치 {weight}는 안에 있는 모든 펀드명에 대해 소숫점 두번째 자리까지 퍼센트로 나타내주세요"
        f"섹터, 펀드명, 가중치 이외의 펀드 정보는 출력하지 말아주시고 특히 {top_fund}는 꼭 따로 언급해주세요"}
    ],
    max_tokens=500,
    temperature=0
    )

    gpt_answer = response.choices[0].message.content.strip()

    # 기본 답변에 나이대별 TDF 추천 연도 추가
    answer = gpt_answer
    if any(sublist[0] == 'TDF' for sublist in sector):
        # 나이대별 TDF 추천 연도 안내 추가
        tdf_recommendations = {
            "20대": "2050-2060",
            "30대": "2040-2050",
            "40대": "2030-2040",
            "50대": "2020-2030"
        }
        tdf_recommendation_text = "추가 안내: 나이대별 TDF 추천 연도는 다음과 같습니다:\n" + \
                                  "\n".join([f"- {age_group}: {years}" for age_group, years in tdf_recommendations.items()])
        answer += "\n\n" + tdf_recommendation_text
    return answer

#################################### 특정 펀드 정보 추출 ####################################
def generate_specific_fund_info_response(user_question, fund_info):
    # 전체 수익률 데이터를 텍스트로 변환
    all_returns_str = ", ".join([
        f"{entry['기준연월']}: {entry['수익률'] * 100:.2f}%" for entry in fund_info.get("펀드 정보", [])
    ]) if fund_info.get("펀드 정보") else "수익률 정보 없음"

    # 프롬프트 생성 - 질문 해석과 필요한 수익률 정보만 출력하도록 요청
    prompt = f"""
        사용자가 요청한 기간에 따라 해당 기간의 수익률 데이터를 제공하세요. 요청에 특정한 기간(예: "최근 6개월", "최근 1년")이 포함된 경우
        그에 맞는 데이터만 반환하세요.질문에 기간이 명시되지 않은 경우, 수익률 정보는 최근 12개월만 제공하세요. \
        기본적으로는 투자목적, 주요 섹터, 국가, 현재일자 기준 최근 12개월 수익률, 투자설명서 및 펀드 정보 다운로드 경로 등의 펀드의 기본 정보를 기본으로 제공합니다.
        질문에 필요한 펀드 정보만 제공해주세요

    펀드의 기본 정보:
    - 투자 목적: {fund_info.get('투자 목적', '정보 없음')}
    - 주요 섹터: {fund_info.get('주요 섹터', '정보 없음')}
    - 국가: {fund_info.get('주요 지역', '정보 없음')}
    - 수익률 정보: {all_returns_str}
    - 투자설명서 및 펀드 정보 다운로드 경로: {fund_info.get('폴더경로', '정보 없음')}
    사용자 질문: "{user_question}"
    """

    # GPT로 문장 생성 요청
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "당신은 펀드 정보를 제공하는 전문가입니다."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0
    )
    gpt_answer = response.choices[0].message.content.strip()

    # user_question에서 기간 정보를 추출하여 그래프 생성
    startdate, enddate = extract_monthly_dates_from_question(user_question)
    graph = generate_price_graph(fund_info, startdate, enddate)

    # 응답과 함께 그래프 이미지 데이터 반환
    return [gpt_answer, graph]

#################################### 최종 응답 ####################################
def chat_with_gpt(user_input, user_profile):
    # GPT에게 질문 분류 요청
    user_grade = user_profile['grade']
    classification = classify_question(user_input)

    if classification == "R":
        sector = classify_sector(user_input, user_profile)
        if sector == 0:
            return "조금 더 정확한 포트폴리오 구성을 위해 섹터를 포함해 질문해 주세요."
        elif sector == '채권':
            return "채권 관련 내용, 재질문"
        else:
            target_fund = recommend_fund(user_profile, sector)

            if target_fund is None:
                return "죄송합니다. 해당 섹터에 대한 펀드를 찾을 수 없습니다."

            cor_funds = find_top_negative_correlations(dff, target_fund,user_profile)
            if cor_funds is None:
                return "죄송합니다. 알맞은 펀드를 찾을 수 없습니다.."
            weight, image = riskfolio(target_fund, cor_funds)
            final_response = generate_final_response(user_grade, sector, target_fund, weight)

            return [final_response, image]

    elif classification == "F":
        fund_name = extract_fund_name(user_input)
        if fund_name.endswith('펀드'):
            fund_name = fund_name[:-2].strip()  # '펀드'를 제거하고 양쪽 공백 제거
        fund_info = get_fund_info(fund_name)
        if fund_info is None:
            return "죄송합니다. 해당 펀드를 찾을 수 없습니다."
        else:
            final_response = generate_specific_fund_info_response(user_input, fund_info)
            return final_response

    else:
        return standard_chatbot(user_input)


##################################################################################################
