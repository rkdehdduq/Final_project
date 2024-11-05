from flask import Flask, request, jsonify, render_template,render_template_string, session
from fundad import chat_with_gpt
from hitmap import map
import os
from dotenv import load_dotenv
from db import dbclient


load_dotenv()

app = Flask(__name__)
app.secret_key = "test_secret_key_1234"  # 실제 배포 환경에서는 복잡한 문자열로 변경 필요

db= dbclient['User']
# 유저 프로필
user_profile = {}

###############################유저 정보 관련###################################################
def get_user_risk_score(username):
    user_data = list(db.user.find({"username": username}, {'_id': 0, 'risk_score': 1}))  # risk_score만 가져오기
    return user_data[0]['risk_score']

def get_user_age(username):
    user_data = list(db.user.find({"username": username}, {'_id': 0, 'age': 1}))  # age만 가져오기
    return user_data[0]['age']

###############################챗봇 출력 시각###################################################
def add_newline_before_bullet_and_last_percent(text):
    # 각 줄의 '-' 기호 앞에 줄바꿈 문자를 추가합니다.
    text = text.replace(' -', '\n -')
    
    # 마지막 '%' 기호의 인덱스를 찾습니다.
    last_percent_index = text.rfind('%')
    
    # 마지막 '%' 기호가 존재하면 그 뒤에 줄바꿈을 추가합니다.
    if last_percent_index != -1:
        text = text[:last_percent_index + 1] + '\n' + text[last_percent_index + 1:]
    return text

###############################투자 성향 측정 기준###################################################
def assess_investment_risk(answers):
    # 나이에 따라 value 값을 설정
    age = answers[0];
    if (age <= 19): 
        answers[0] = 0;
    elif (age >= 20 and age <= 40):
        answers[0] = 5;
    elif (age >= 41 and age <= 50):
        answers[0] = 10;
    elif (age >= 51 and age <= 60):
        answers[0] = 15;
    elif (age >= 61):
        answers[0] = 20;
    
    score = sum(answers)
    
    if score <= 20:
        return score, "안정형 (Stable)"
    elif score <= 40:
        return score, "안정추구형 (Stable Seeking)"
    elif score <= 60:
        return score, "위험 중립형 (Risk Neutral)"
    elif score <= 80:
        return score, "적극 투자형 (Aggressive Investor)"
    else:
        return score, "공격투자형 (Aggressive Investor)"
    
##################################################################################

###############################첫페이지###################################################
@app.route('/')
def home():
    return render_template('login.html')

###############################메인페이지###################################################
@app.route('/index')
def index():
    return render_template('index.html')  # index.html 페이지

###############################히트맵 라우팅###################################################
@app.route('/hitmap')
def hitmap_route():
    return render_template_string(map)  # map 리스트를 템플릿에 전달


###############################챗봇과 대화###################################################
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('input')
    user_profile = session.get('user_profile')

    result = chat_with_gpt(user_input, user_profile)
    print(user_profile.get('grade'))  # 추가된 부분: user_profile의 grade를 콘솔에 출력

    if type(result) == list:
        # formatted_text = add_newline_before_bullet_and_last_percent(result[0])
        # result[0] = formatted_text
        print(type(result))
        # JSON 형태로 결과를 반환
        return jsonify(result)
    else:
        return jsonify(result)

###############################signin.html 페이지 렌더링###################################################
@app.route('/signin')
def signin():
    return render_template('signin.html')  

###############################투자 성향 파악###################################################
@app.route('/test', methods=['GET', 'POST'])
def test():
    if request.method == "POST":
        answers = [int(request.form[f'question{i}']) for i in range(1, 8)]
        score, riskLevelText = assess_investment_risk(answers)
        return jsonify(score=score, riskLevelText=riskLevelText)  # JSON 응답
    return render_template("test.html")


###############################재검사 하기###################################################
@app.route('/test2', methods=['GET', 'POST'])
def test2():
    return render_template('test2.html')  # test2.html 페이지 반환

###############################재검사 후 db에 넣기###################################################
@app.route('/re_test', methods=['GET','POST'])
def re_test():
    if request.method == "POST":
        answers = [int(request.form[f'question{i}']) for i in range(1, 8)]
        score, riskLevelText = assess_investment_risk(answers)
        return jsonify(score=score, riskLevelText=riskLevelText)  # JSON 응답

###############################재검사 결과 저장 API###################################################
@app.route('/api/save_retest_result', methods=['POST'])
def save_result():
    data = request.get_json()  # JSON 형식의 데이터 수신
    risk_level = data.get('riskLevel')  # 위험 수준
    riskLevelText = data.get('riskLevelText')  # 위험수준 텍스트
    user_id = data.get('username')
    user_profile['grade']=risk_level
    age = data.get('age')
    print(user_profile.get('grade'))
    # 사용자 정보를 업데이트
    try:
        result = db.user.update_one(
            {'username': user_id},  # 사용자 ID로 찾기
            {'$set': {'risk_score': risk_level, 'riskLevelText': riskLevelText, 'age': age}}  # 위험 수준과 점수 업데이트
        )
        if result.matched_count == 0:
            print(f'사용자를 찾을 수 없습니다: {user_id}')
        elif result.modified_count == 0:
            print('업데이트할 데이터가 없습니다.')
        return jsonify({'message': '결과가 성공적으로 저장되었습니다.', 'closePopup': True}), 200  # 성공 응답
    except Exception as e:
        print('결과 저장 중 오류 발생', e)  # 오류 메시지 출력
        return jsonify({'error': '결과 저장 중 오류가 발생했습니다.'}), 500  # 오류 응답


###############################회원가입 API###################################################
@app.route('/api/signup', methods=['POST'])
def signup():
    # 클라이언트로부터 JSON 형식의 데이터 수신
    data = request.get_json()
    fullname = data.get('fullname')  # 이름
    email = data.get('email')        # 이메일
    username = data.get('username')  # 사용자 이름
    password = data.get('password')  # 비밀번호
    risk_score = data.get('riskLevel')  # 위험 점수 가져옴 (변경된 부분)
    riskLevelText = data.get('riskLevelText')
    age = data.get('age')
    # 입력값 확인: 모든 필드가 채워져 있는지 확인
    if not fullname or not email or not username or not password or riskLevelText is None:
        return jsonify({'error': '모든 필드를 입력해야 합니다.'}), 400
    try:
        # 사용자 정보를 딕셔너리 형태로 저장
        new_user = {
            'fullname': fullname,
            'email': email,
            'username': username,
            'password': password,  # 비밀번호 암호화 필요
            'risk_score': risk_score,  # 위험 점수 저장
            'riskLevelText':riskLevelText,
            'age': age
        }
        db.user.insert_one(new_user)  # MongoDB에 사용자 정보 저장
        return jsonify({'message': '사용자가 성공적으로 생성되었습니다.'}), 201  # 성공 응답
    except Exception as e:
        print('사용자 생성 중 오류 발생', e)  # 오류 메시지 출력
        return jsonify({'error': '사용자를 생성하는 동안 오류가 발생했습니다.'}), 500  # 오류 응답


###############################로그인 API###################################################
@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    # 입력값 확인
    if not username or not password:
        return jsonify({'error': '사용자 이름과 비밀번호를 입력해야 합니다.'}), 400
    try:
        # 사용자 조회
        user = db.user.find_one({'username': username})
        if not user or user['password'] != password:
            return jsonify({'error': '사용자 이름 또는 비밀번호가 잘못되었습니다.'}), 401
        # 로그인 성공
        return jsonify({'message': '로그인 성공'}), 200
    except Exception as e:
        print('로그인 중 오류 발생', e)
        return jsonify({'error': '로그인 처리 중 오류가 발생했습니다.'}), 500


###############################계정 정보 API###################################################
@app.route('/api/account', methods=['GET'])
def get_account_info():
    username = request.args.get('username')  # 쿼리 파라미터에서 사용자 이름 가져오기
    if not username:
        return jsonify({'error': '사용자 이름이 필요합니다.'}), 400
    user = db.user.find_one({'username': username}, {'_id': 0, 'username': 1, 'email': 1, 'riskLevelText': 1})
    if user:
        return jsonify(user), 200  # 사용자 정보 반환
    else:
        return jsonify({'error': '사용자를 찾을 수 없습니다.'}), 404  # 사용자 없음

###############################유저 이름 받아오기###################################################
@app.route('/api/username', methods=['POST'])
def receive_username():
    data = request.get_json()
    username = str(data.get('username'))
    print(f"Received username: {username}")
    # 사용자 이름을 사용하여 risk_score를 가져오기
    risk_score = get_user_risk_score(username)
    age = int(get_user_age(username))
    # 세션에 user_profile 저장
    session['user_profile'] = {
        "username": username,
        "grade": risk_score,
        "age": age
    }
    
    # 세션 저장 상태 확인
    session.modified = True
    if risk_score is not None:
        print(f"Risk score for user '{username}': {risk_score}")
        return jsonify({"message": "Username received successfully."}), 200
    else:
        print("Failed to retrieve risk score.")

if __name__ == '__main__':
    app.run()
