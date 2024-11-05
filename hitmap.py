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
import kaleido # import the kaleido package


# 폰트 설정
matplotlib.rcParams['font.family'] = 'Malgun Gothic'


db = dbclient['fund_data']  # 데이터베이스 이름

## 트리맵 디비
collection = db['fund_info']
data_f = list(collection.find())
dff = pd.DataFrame(data_f)

## 히트맵 디비
collectiong = db['fund_graph']
data_g = list(collectiong.find())
dfg = pd.DataFrame(data_g)


################## - 트리맵 데이터 #################################### 
# 주요 섹터 컬럼에서 NaN 값을 빈 리스트로 대체
dff['주요 섹터'] = dff['주요 섹터'].apply(lambda x: x if isinstance(x, list) else [])

# 주요 섹터 컬럼에서 섹터와 섹터2를 분리
dff[['섹터', '섹터2']] = pd.DataFrame(dff['주요 섹터'].tolist(), index=dff.index)

# 섹터2가 없는 경우 섹터를 섹터2로 복사
dff['섹터2'] = dff.apply(lambda row: row['섹터2'] if pd.notna(row['섹터2']) else row['섹터'], axis=1)

############################## - 히트맵 데이터 ################################
# returns 컬럼을 연도별로 확장하여 각 연도가 별도 컬럼이 되도록 변환
df_expanded = pd.DataFrame(dfg["returns"].tolist(), index=dfg["sector"])
# 색상 팔레트를 생성하고 섹터마다 고유 색상을 할당
unique_sectors = dfg['sector'].unique()
colors = sns.color_palette("tab20", len(unique_sectors))
color_map = dict(zip(unique_sectors, colors))

#######################-  트리맵 함수 ###########################
# 1. 최신 데이터 가져오기 함수 (num_entries 개수만큼)
def get_latest_data(fund_info, num_entries):
    return fund_info[-num_entries:] if len(fund_info) >= num_entries else fund_info

# grouped_data 준비 함수
def prepare_grouped_data(dff, num_entries):
    filtered_entries = []

    for _, row in dff.iterrows():
        # 최신 num_entries 개의 데이터 추출
        latest_data = get_latest_data(row['펀드 정보'], num_entries * 2)

        # 최신 기간과 이전 기간으로 나누기
        recent_data = latest_data[:num_entries]  # 최신 기간
        previous_data = latest_data[num_entries:num_entries * 2]  # 이전 기간

        # 최신 기간 규모 및 수익률 데이터 계산
        recent_total_size = sum(data['기준가격'] * data['설정원본'] for data in recent_data)
        previous_total_size = sum(data['기준가격'] * data['설정원본'] for data in previous_data)

        # 설정액 증감 계산 (이전 평균 대비 최신 평균의 증감)
        size_change = (recent_total_size / num_entries) - (previous_total_size / num_entries) if previous_total_size else 0

        # 데이터 정리하여 각 row에 추가
        for data in recent_data:
            filtered_entries.append({
                '섹터': row['섹터'],
                '섹터2': row['섹터2'],
                '기준연월': data['기준연월'],
                '규모': data['기준가격'] * data['설정원본'],
                '수익률': data.get('수익률', 0),
                '설정액 증감': size_change  # 증감 추가
            })
    # DataFrame으로 변환 후 월별로 그룹화하여 섹터 규모 및 수익률 평균 계산
    period_df = pd.DataFrame(filtered_entries)
    period_df['기준연월'] = pd.to_datetime(period_df['기준연월'], format='%Y-%m')

    monthly_data = period_df.groupby(['섹터', '섹터2', period_df['기준연월'].dt.to_period('M')]).agg({
        '규모': 'sum',
        '수익률': 'mean',
        '설정액 증감': 'first'  # 증감 값 유지
    }).reset_index()

    # 전체 섹터별로 평균 규모 및 수익률 계산
    grouped_data = monthly_data.groupby(['섹터', '섹터2']).agg({
        '규모': 'mean',
        '수익률': 'mean',
        '설정액 증감': 'first'  # 섹터별 증감 평균
    }).reset_index()
    grouped_data['수익률'] = grouped_data['수익률'].fillna(0)

    return grouped_data

# 2. 트리맵 생성 함수
def create_treemap(grouped_data,title):
    # '설정액 증감'을 계산하여 증감 표시를 추가
    grouped_data['증감 표시'] = grouped_data['설정액 증감'].apply(lambda x: '▲' if pd.notna(x) and x > 0 else '▼' if pd.notna(x) and x < 0 else '')  # 증감에 따라 삼각형 표시

    fig = px.treemap(
        grouped_data,
        path=['섹터', '섹터2'],  # 섹터와 섹터2를 계층으로 사용
        values='규모',
        color='수익률',
        color_continuous_scale=[[0, 'blue'], [0.5, 'white'], [1, 'red']],
        color_continuous_midpoint=0,  # 0을 중간값으로 설정하여 양수와 음수가 구분되도록 함
        range_color=[-0.04, 0.04],  # 색상 범위를 -5%에서 +5%로 설정
        hover_data={
            '수익률': ':.2%',  # 수익률 정보 표시
            '설정액 증감': ':.2f',  # 설정액 증감 정보 표시
            '증감 표시': True  # 증감 표시도 추가
        },
        title=title
    )

    # 텍스트 설정 및 hovertemplate으로 설정 (증감 표시가 먼저, 설정액 증감이 두 번째로 표시)
    fig.update_traces(
        texttemplate="%{label}<br>%{customdata[2]} %{customdata[1]:,.0f}<br>%{customdata[0]:.2%}",  # 증감 표시, 설정액 증감 및 수익률 정보 표시
        textfont_size=16,  # 텍스트 폰트 크기 설정
        textinfo="label+text",  # 텍스트 정보를 라벨과 함께 표시
        hovertemplate="<b>%{label}</b><br>설정액 증감: %{customdata[1]:,.0f}<br>수익률: %{color:.2%}",  # hover 시에 정보 순서 변경
    )

    # 레이아웃 설정
    fig.update_layout(
        margin=dict(t=50, l=25, r=25, b=25),
        title_font_size=24  # 제목 폰트 크기 설정
    )
    return fig
##################################################################################################

##########################################트리맵 함수 ########################################################
# 연도별 상위 10개 섹터 추출
def get_top_sectors(num):
    top_sectors_by_year = {}
    for year in df_expanded.columns:
        top_sectors_by_year[year] = df_expanded[year].nlargest(num)
    return top_sectors_by_year

def split_text_br(text):
    if "-" in text:
        parts = text.split("-")
        return parts[0].strip() + "<br>" + parts[1].strip()
    elif text == "ETF Managed Portfolio":
        return "ETF Managed<br>Portfolio"
    elif text == "지수연계(ELS,ELB)":
        return "지수연계<br>(ELS,ELB)"
    elif len(text) > 10:
        return text[:10] + "<br>" + text[10:]
    else:
        return text

def top_sectors_plot_interactive(num):
    # 상위 섹터를 가져오고 색상 팔레트를 생성
    top_sectors_by_year = get_top_sectors(num)
    years = sorted(top_sectors_by_year.keys())
    unique_sectors = set(sector for sectors in top_sectors_by_year.values() for sector in sectors.keys())

    # 고유 섹터마다 색상을 할당
    colors_pastel1 = sns.color_palette("Pastel1", 9).as_hex()
    colors_set3 = sns.color_palette("Set3", 12).as_hex()    # 12가지 색상 (파스텔 톤)

    # 두 팔레트를 결합하여 전체 색상 팔레트 생성
    colors_combined = colors_set3 + colors_pastel1
    color_map = dict(zip(unique_sectors, colors_combined * (len(unique_sectors) // len(colors_combined) + 1)))

    marker_size = 60
    map_height = marker_size * (num+1.5)  # 높이만 동적으로 설정하여 rank가 많아질 때만 확장

    fig = go.Figure()

    # 각 연도별로 상위 섹터를 개별 타일로 추가
    for i, year in enumerate(years):
        top_sectors = top_sectors_by_year[year]

        for rank, (sector, return_value) in enumerate(top_sectors.items()):
            fig.add_trace(go.Scatter(
                x=[year],
                y=[rank + 1],
                mode="markers+text",
                marker=dict(
                    color=color_map[sector],
                    size=marker_size,  # 사각형 크기 조정
                    symbol="square"
                ),
                text=split_text_br(sector) + f"<br>{round(return_value, 1)}%",  # 두 줄로 나눈 텍스트 적용
                textposition="middle center",
                hovertemplate=f"Year: {year}<br>Sector: {sector}<br>Return: {round(return_value, 1)}%",
                name="",  # trace 이름 비우기
                showlegend=False
            ))

    # 레이아웃 설정
    fig.update_layout(
        title=f"연도별 섹터 수익률 Top 10",
        title_font_size=24,
        xaxis=dict(
            title="Year",
            tickmode="array",
            tickvals=years,
            tickangle=45,
            side="top",  # 연도 라벨을 상단으로 이동
            title_standoff=10,  # 타이틀을 아래로 10px 내림
            ticklabelposition="outside top",  # 라벨을 축 밖 상단에 배치
            range=[-0.5, len(years) - 0.5],
            showgrid=False  # x축 격s
        ),
        yaxis=dict(
            title="Rank",
            tickvals=list(range(1, num+1)),
            autorange="reversed",
            showgrid=False,  # y축 격자 제거

        ),
        plot_bgcolor="white",  # 배경 흰색 설정
        height=map_height,  # 전체 그래프 크기 조정
        
        showlegend=False,
        margin=dict(l=0, r=0, t=100, b=0),  # 여백 최소화
        )
    return fig

##################################################################################################

# ##########################################히트+트리 맵 html 생성########################################################
def create_combined_treemap_html(fig1, fig2, fig3):
    # 각 fig를 HTML 문자열로 변환합니다.
    html_str1 = fig1.to_html(full_html=False, include_plotlyjs='cdn')
    html_str2 = fig2.to_html(full_html=False, include_plotlyjs=False)
    html_str3 = fig3.to_html(full_html=False, include_plotlyjs=False)
    # 두 HTML 문자열을 결합하여 하나의 HTML 파일로 생성합니다.
    combined_html = f"""
    <html>
        <head>
            <title>Combined Treemap</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
            
            {html_str1}
            {html_str2}
            {html_str3}
        </body>
    </html>
    """

    return combined_html
# 1년 등 각 기간에 따라 데이터를 준비합니다.
grouped_data_1year = prepare_grouped_data(dff, num_entries=12)  # 1년
small_sectors_1year = grouped_data_1year.loc[grouped_data_1year['규모'] < grouped_data_1year['규모'].quantile(0.4)].copy()
# 트리맵 생성
one_year_treemap = create_treemap(grouped_data_1year,title="최근 1년간 섹터별 수익률 트리맵(크기-설정액, 색상-수익률)")
small_sectors_treemap = create_treemap(small_sectors_1year, title="- 하위 40퍼센트")
hitmap =top_sectors_plot_interactive(10)

map= create_combined_treemap_html(one_year_treemap, small_sectors_treemap, hitmap)