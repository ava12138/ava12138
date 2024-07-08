from langchain.chains import LLMChain
from langchain.prompts.chat import ChatPromptTemplate
from langchain.prompts import PromptTemplate  
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Tongyi
import time
import os
from tqdm import tqdm
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import numpy as np
import streamlit as st

os.environ["DASHSCOPE_API_KEY"] = "sk-929f75f6b8af4c0db579490dc255aa85"

template = "你是一个教育出题专家,擅长根据题目和题目要求（包含正确答案）设置3个干扰项，干扰项的设置原则为:{principle}。请根据以下格式提供答案选项：\nA.\nB.\nC.\nD.请确保你的答案选项遵循上述格式，并且包含一个正确答案和三个干扰项。你不需要给出解释，只需要提供四个选项即可。"
topic_template = "这是题目和题目要求，括号中是正确答案：{topic}"

output_parser = StrOutputParser()


# 示例
topic = "唐代中后期文人间流行诗歌唱和之风，“江南”成为唱和的重要主题。苏州、杭州、湖州、宣州（今安徽宣城）等地名经常在唱和诗歌中出现。这种风尚：（正确答案：得益于稳定的地方秩序）"
# topic = "1960~1970年，发展中国家对发达资本主义国家的出口额从197.8亿美元增加到397.5亿美元，从发达资本主义国家的进口额也从218亿美元增加到413.6亿美元。在国际贸易中，发展中国家出口额比重从21.4%下降至17.6%.据此可知，该时期：（正确答案：发展中国家经济地位下降）"
principles = {
    '混淆相似概念': '使用看似相关但实际不正确的信息，以测试考生对细节的注意力和记忆的准确性。',
    '概念替换': '交换题干中的核心概念，或使用相似概念替换，以检验考生对主题的深度理解。',
    '逆向思维': '通过反转题干的预期逻辑，设置陷阱，考察考生是否能逆向操作找出正确答案。',
    '主次关系颠倒': '改变概念间的主次、因果等逻辑关系，考验考生对事物内在联系的判断。',
    '过度细化或泛化': '提供细节过多或过于泛化的选项，迷惑考生，使其难以把握问题的核心。',
    '答非所问': '引入与题干无关但表面看似正确的信息，诱使考生偏离题目的真正焦点。'
}

def generate_responses_from_tongyi(topic: str, principles_select: str, numberOfAnswer: int, temperature: float, scores=None) -> list[str]:
    feedback = ""
    print(scores)
    print(temperature)
    if scores:
      feedback = "\n你的回答质量由分数决定，上一次的回答分数为：{scores}.你需要根据该回答分数来判断如何优化回答。分数小于0需要完全重新回答，大于0则在当前基础上增加回答的专业程度。再次强调，你不需要生成解释，只需要提供选项。 "
    current_template = template + feedback
    llm = Tongyi(model_name="qwen-max", temperature=temperature)  
    prompt = ChatPromptTemplate.from_messages([
      ("system", current_template),
      ("user", topic_template)
    ])
    
    print(current_template)
    chain = prompt | llm | output_parser
    generated_responses = []
    start_time = time.time()
    combined_principle_description = "；".join(
                [f"原则{i+1}：{principle} - {principles[principle]}" for i, principle in enumerate(principles_select)]
            )
    with st.spinner('加载中...'):
      # 生成多个回答
      if scores is None:
        with tqdm(total=numberOfAnswer, desc="Generating responses") as pbar:
            for _ in range(numberOfAnswer):
                response = chain.invoke({"topic": topic, "principle": combined_principle_description})
                generated_responses.append(response)
                pbar.update(1)
      else:
        with tqdm(total=numberOfAnswer, desc="Generating responses") as pbar:
            for i in range(numberOfAnswer):
                response = chain.invoke({"topic": topic, "principle": combined_principle_description, "scores": scores[i]})
                generated_responses.append(response)
                pbar.update(1)
    return generated_responses, time.time() - start_time

def calculate_information_score(response: str) -> float:
    
    cleaned_response = re.sub(r'[^\w\s]', '', response).lower()
    
    words = cleaned_response.split()
    unique_words_count = len(set(words))
    return unique_words_count

def adjust_scores_based_on_similarity(responses: list[str], base_scores: list[float]) -> list[float]:
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(responses)
    cosine_sim_matrix = cosine_similarity(tfidf_matrix)
    similarity_scores = [0] * len(responses)
    for i in range(len(responses)):
        similarity_scores[i] = max([cosine_sim_matrix[i][j] for j in range(len(responses)) if i != j])
    adjusted_scores = [base - (sim * 10) for base, sim in zip(base_scores, similarity_scores)]
    adjusted_scores = [max(0, score) for score in adjusted_scores]
    return adjusted_scores
    
def generate_quality_scores(sorted_indices: list[int], responses: list[str]) -> list[float]:
    # 基于排序信息计算基础分数
    base_scores = [len(sorted_indices) - sorted_indices.index(i) for i in range(1, len(sorted_indices) + 1)]
    
    # 根据长度调整分数
    length_scores = [len(response) for response in responses]
    max_length = max(length_scores)
    length_scores = [score / max_length * 10 for score in length_scores]  
    
    # 计算信息丰富性分数
    information_scores = [calculate_information_score(response) for response in responses]
    max_information_score = max(information_scores)
    information_scores = [score / max_information_score * 10 for score in information_scores]
      
    # 计算信息相似性分数
    similarity_adjusted_scores = adjust_scores_based_on_similarity(responses, base_scores)
    
    combined_scores = [sim_adj + length + info for sim_adj, length, info in zip(similarity_adjusted_scores, length_scores, information_scores)]
    
    # 使用Z-score标准化
    mean_score = np.mean(combined_scores)
    std_deviation = np.std(combined_scores)
    if std_deviation == 0:
        # 避免除以零
        normalized_scores = [0 for _ in combined_scores]
    else:
        normalized_scores = [(score - mean_score) / std_deviation for score in combined_scores]
    
    return normalized_scores

def display_response(responses):
    st.caption(
        """<p align="left">模型回答如下：</p>""",
        unsafe_allow_html=True,
    )
    num_responses = len(responses)
    cols = st.columns(num_responses)  # 创建列的数量与回答数量相同
    for i, (col, response) in enumerate(zip(cols, responses), start=1):
        with col:
            st.markdown(f"**回答 {i}：**")
            formatted_response = response.replace('B.', '\nB.').replace('C.', '\nC.').replace('D.', '\nD.')
            st.write(formatted_response)
            st.write("")  # 添加一个空行以增加可读性
    
def adjust_temperature_based_on_rank(rank_info, current_temperature, base_adjustment_factor=0.1, min_temperature=0.1, max_temperature=1.0):
    best_rank = int(rank_info[0])
    # 动态调整调整因子，排名越靠前，调整幅度越大
    adjusted_factor = base_adjustment_factor * (len(rank_info) - best_rank) / (len(rank_info) - 1)
    # 限制调整因子在合理范围内
    adjusted_factor = min(max(adjusted_factor, 0), 1 - abs(current_temperature - min_temperature), 1 - abs(current_temperature - max_temperature))
    if best_rank <= len(rank_info) / 2:
        # 下降温度
        new_temperature = max(min_temperature, current_temperature - adjusted_factor * (current_temperature - min_temperature))
    else:
        # 上升温度
        new_temperature = min(max_temperature, current_temperature + adjusted_factor * (max_temperature - current_temperature))
    return new_temperature

# 使用示例


def get_response(current_temperature, topic, principle, scores):
    numberOfAnswer = 4
    responses, elapsed_time = generate_responses_from_tongyi(topic, principle, numberOfAnswer, current_temperature, scores)
    st.caption(
        f"""<p align="left">生成回答耗时 {elapsed_time:.2f} 秒</p>""",
        unsafe_allow_html=True,
    )
    return responses
 
def main():
    st.set_page_config(page_title="干扰项设置与反馈系统", layout="wide")
    st.title("基于大模型的干扰项设置与反馈系统")
    st.markdown("""
    <style>
        .main {background-color: #ffffff; padding: 20px;}
        .stButton > button{
            background-color: #40bb45; 
            color: white;
        }
        .highlighted-text {
            background-color: #007ACC;  
            color: white;
            font-size: 1.2em;
            font-weight: bold;
            padding: 10px;
            border-radius: 5px;
        }
        .important-caption {
            font-size: 18px; /* 增大字体大小 */
            font-weight: bold; /* 加粗文本 */
            color: #DD6655; /* 改变文本颜色 */
            background-color: #FFF8F2; /* 添加背景色 */
            border: 1px solid #DD6655; /* 添加边框 */
            border-radius: 5px; /* 边框圆角 */
            padding: 10px; /* 添加内边距 */
            margin: 10px 0; /* 上下外边距 */
        }
    </style>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="highlighted-text">
    唐代中后期文人间流行诗歌唱和之风，“江南”成为唱和的重要主题。苏州、杭州、湖州、宣州（今安徽宣城）等地名经常在唱和诗歌中出现。<br>这种风尚：
    </div>
    """, unsafe_allow_html=True)
    # 初始温度
    if 'current_temperature' not in st.session_state:
        st.session_state.current_temperature = 0.7

    if 'scores' not in st.session_state:
        st.session_state.scores = None

    if 'responses' not in st.session_state:
        st.session_state.responses = None

    if 'principle' not in st.session_state:
        st.session_state.principle = []
    
    # 提交按钮状态
    if 'submitted' not in st.session_state:
        st.session_state.submitted = False

    def submit_form():
        st.session_state.submitted = True

    def reset_form():
        st.session_state.submitted = False

    with st.form("input_form"):
        col1, col2 = st.columns([2, 1])
        with col1:
            principle = st.multiselect(
                label='请选择干扰项原则',
                options=list(principles.keys()),
                format_func=str,
                key="principle_multiselect"
            )
        with col2:
            pass
            
        if st.session_state.submitted:
            # 保存用户选择的principle
            st.session_state.principle = principle
            # 获取并显示回答
            st.session_state.responses = get_response(st.session_state.current_temperature, topic, principle, st.session_state.scores)
            temp = st.session_state.responses
            display_response(temp)
        
        submitted = st.form_submit_button("提交并生成回答", on_click=submit_form)
            
            
    if st.session_state.responses:
        rank_info = st.text_input('请输入排序（如：4321）',  max_chars=20, key='rank_input')
        if rank_info.strip():
            rank_list = [int(rank) for rank in rank_info.split()]
            st.session_state.scores = generate_quality_scores(rank_list, st.session_state.responses)
            st.session_state.current_temperature = adjust_temperature_based_on_rank(rank_info, st.session_state.current_temperature)

    exit_button = st.button("结束", type="primary", on_click=reset_form)


    if exit_button:
        st.markdown("""
        <div class="important-caption">
        您已选择退出，服务已终止。
        </div>
        """, unsafe_allow_html=True)
        st.stop()

if __name__ == "__main__":
    main() 
              