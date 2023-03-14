# -*- coding: utf8 -*-

import streamlit as st
import streamlit.components.v1 as components

from PIL import Image
import pandas as pd
import numpy as np
import networkx as nx
import json
import pickle as pkl
import time
import sys
import os, glob

import warnings
warnings.filterwarnings('ignore') #경고 무시용

import sys
sys.path.append('/home/agens/conda_user/scene/aivg/streamlit/pages')
from utils.scene_graph import *

st.set_page_config(layout="wide")

st.markdown("""
<style>
.center {
    text-align: center; 
    color: Black;
    font-size:150% !important;
}
.block-container {
                    padding-top: 1rem;
                    padding-bottom: 0rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
""", unsafe_allow_html=True)

with open(file='/home/agens/conda_user/scene/aivg/streamlit/data/tbl_scene.pkl', mode='rb') as f:
    tbl_scene=pkl.load(f)

def get_spo(Subject, Predicate, Object):
    condition = ((tbl_scene.subject == Subject)&(tbl_scene.predicate == Predicate)&(tbl_scene.object == Object))
    img_lst=list(tbl_scene.loc[condition]['image_id'].values[:10])
    groups=tbl_scene.loc[tbl_scene['image_id'].isin(img_lst)].groupby('image_id')            
    return  dict(list(groups))

def image_resize(image_file,width=200,height=300):
    image=Image.open(image_file)
    new_image = image.resize((width,height))
    return new_image


def load_image(img_path):
    img = Image.open(img_path)
    return img


def Intro():

    st.markdown("### <h1 style='text-align: center; color: Black; font-size:350%'>Scene Graph란?</h1>", unsafe_allow_html=True)

    det_exp = """
                Scene Graph(장면 그래프)란 이미지 및 영상 데이터의 <strong>장면(scene)에서 객체(object) 및 관계(relationship)를 추출</strong>하고 이를 
                <strong>주어-술어-목적어' 관계인 SPO(Subject, Predicate, Object) 형태로 그래프를 통해 표현</strong>하는 방법입니다.<br><br>
                &nbsp본 R&D는 기존 장면의 객체들 간의 관계의 SPO를 
                <strong>그래프 데이터 베이스(GDB)</strong>를 이용한 <strong>지식그래프(Knowledge Graph)</strong>로 표현하여, 관계기반 데이터의 조회 및 추출을 편리하게하고
                더 나아가 이미지나 영상의 장면 유사도, 예측 그리고 자동화 시스템을 만드는 것을 목표로 합니다."""
    
    det_exp_font = f"""<h6 style='text-align: left; color: #1b1b1b; font-family : times arial; 
    line-height : 165%; font-size : 117%; font-weight : 400'>{det_exp}\n\n</h6>"""


    st.markdown("#### <h1 style='text-align: left; color: #565656; font-size:230%'>Intro</h1>", unsafe_allow_html=True)
    st.write("")
    
    col1,col2 = st.columns([8.5, 1])

    with col1 :
        st.markdown(det_exp_font, unsafe_allow_html=True)
        st.write("")



def Explanation():

    st.markdown("### <h1 style='text-align: center; color: Black; font-size:270%'>그래프 데이터베이스(GDB)를 이용한 Scene Graph</h1>", unsafe_allow_html=True)
    
    det_exp = """
                &nbsp이번 챕터에는 <strong>그래프 데이터 베이스(GDB)</strong>를 사용하여 
                Scene Graph를 <strong>지식그래프(Knowledge Graph) 형태</strong>로 표현합니다.<br><br>
                &nbspScene Graph를 GDB의 강점인 <strong>속성(property)</strong>정보를 이용한 
                LPG(Labeled Property Graph) 형태로 모델링하여 단어 간의 SPO의 관계를 유연하게 설명하고, 다양한 그래프 알고리즘 등을 사용할 수 있습니다."""
    
    det_exp_font = f"""<h6 style='text-align: left; color: #1b1b1b; font-family : times arial; 
    line-height : 165%; font-size : 117%; font-weight : 400'>{det_exp}\n\n</h6>"""


    st.markdown("#### <h1 style='text-align: left; color: #565656; font-size:230%'>Explanation</h1>", unsafe_allow_html=True)
    st.write("")
    
    col1,col2 = st.columns([8.5,1])

    with col1 :
        st.markdown(det_exp_font, unsafe_allow_html=True)
        st.write("")
    st.markdown("___")
#----------------------------------------------------------------------------------------------------------------
    st.markdown("""## <h1 style='text-align: left; color: #3b3b3b; font-size:180%'>☑ GDB를 이용한 Scene Graph 모델링</h1>""", unsafe_allow_html=True)
    img_2 = '/home/agens/conda_user/scene/aivg/streamlit_img/part1_img2.PNG'
    img2 = load_image(img_2)
    text2 = """
            Scene Graph의 모델링 방법론은 ⑴술어(predicate)를 하나의 노드로 따로 표현하는 방식과 ⑵술어를 엣지로써 표현하는 두 가지 방식이 있습니다.\n
            두 그래프 모델링 방법과 지식그래프화를 하기 위해 어떤 것이 효율적인지에 대해 설명합니다."""
    st.write(text2)

    st.text("")
    st.image(img2, width=1000)
    st.text("")

    img2_text = """
            <strong style="color:#515151; font-size : 120%">• 술어(predicate)를 <i>노드(node)</i>로 그래프 모델링하는 경우</strong><br>
            &nbsp&nbsp&nbsp&nbsp◦ 일반적으로 Scene Graph 모델링 시 속성(property)정보가 없는 RDF(Resource Description Framework) 
            형태의<br>
            &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp그래프로써 가장 많이 사용하는 기법<br>
            &nbsp&nbsp&nbsp&nbsp◦ 술어를 노드로 표현하여 직관적일 수 있지만, <i>노드의 수가 많아져서 연산량 증가</i><br><br>
            <strong style="color:#515151; font-size : 120%">• 술어(predicate)를 <i>엣지(edge)</i>로 그래프 모델링하는 경우</strong><br>
            &nbsp&nbsp&nbsp&nbsp◦ 객체 간 관계를 나타내는 <strong>술어를 엣지</strong>로 표현하여 기존 SPO의 형태를 유연하게 나타낸 그래프 모델링<br>
            &nbsp&nbsp&nbsp&nbsp◦ GDB의 <i>속성정보를 이용</i>하여 술어에 대한 부가 정보를 엣지의 속성정보로 삽입 가능<br>
            &nbsp&nbsp&nbsp&nbsp◦ 술어를 엣지로 표현하여 노드로 표현했을 때보다 <strong>적은 용량의 DB모델 및 연산속도 감소</strong>"""
    #st.wrtie(img1_text)
    img2_text_html = f"""
    <h6 style='text-align: left;
    color: #Black; font-family : times arial; line-height : 200%; 
    font-size : 100%; font-weight : 500'>{img2_text}\n\n</h6>"""
    st.markdown(img2_text_html, unsafe_allow_html=True)
    st.text("")
#----------------------------------------------------------------------------------------------------------------

def Practice1():

    ttl_txt1_1 = "☑ 이미지를 이용한 및 SPO Scene Graph 추출"
    st.markdown(f"""## <h1 style='text-align: center; color: #3b3b3b; font-size:150%'>{ttl_txt1_1}</h1>""", 
    unsafe_allow_html=True)
    st.text("")
    st.write("원하시는 이미지를 클릭하면, 아래 Scene Graph가 출력됩니다.")
    
    imageCarouselComponent = components.declare_component("image-carousel-component", path="/home/agens/conda_user/scene/aivg/streamlit/Streamlit-Image-Carousel/frontend/public")
    
    imageUrls = [ 
        "https://i.ibb.co/y0dHF08/new-2320618.jpg",
        "https://i.ibb.co/s9QjfDz/new-2335472.jpg",
        "https://i.ibb.co/5swc4Tt/new-2343076.jpg",
        "https://i.ibb.co/c2SNbyN/new-2366051.jpg",
        "https://i.ibb.co/QPmHVKm/new-2374468.jpg",
        "https://i.ibb.co/K0YRXMn/2319903.jpg",
        "https://i.ibb.co/d6Mj9Dm/2355755.jpg",
        "https://i.ibb.co/q5wkJGg/2367614.jpg",
        "https://i.ibb.co/jRLww69/2368398.jpg"
        ]
    df=pd.read_csv('/home/agens/conda_user/scene/aivg/streamlit/data/img_spo_10.csv')
    groups=df.groupby('image_id')
    df_dict = dict(list(groups))
    selectedImageUrl = imageCarouselComponent(imageUrls=imageUrls, height=200)

    st.markdown("___")

    if selectedImageUrl is not None:
        img_idx = int(selectedImageUrl[selectedImageUrl.index('.jpg')-7:selectedImageUrl.index('.jpg')])
        st.text(f"• 이미지 번호 {img_idx}를 선택하셨습니다.")


        col_1, col_2, col_3, col_4 = st.columns([3.5, 0.2, 7.1, 0.2])


        with col_1:
            
            if selectedImageUrl is not None:

                min_ttl1 = f"Result 1 : selected <i>Image</i>"
                st.markdown(f"""## <h5 style='text-align: center; color: #3b3b3b; font-size:250%, font-weight = 600'>{min_ttl1}</h5>""", 
                        unsafe_allow_html=True)
                
                with st.spinner('Loading for Scene Graph...⌛️'):
                    time.sleep(0.5)

                    st.image(selectedImageUrl, width = 500)
                    #img_idx = int(selectedImageUrl[selectedImageUrl.index('.jpg')-7:selectedImageUrl.index('.jpg')])
                    
                
        with col_3:

            min_ttl2 = f"Result 2 : <i>Scene Graph</i> of selected Image"
            st.markdown(f"""## <h5 style='text-align: center; color: #3b3b3b; font-size:250%, font-weight = 100'>{min_ttl2}</h5>""", 
                        unsafe_allow_html=True)

            if selectedImageUrl is not None:

                graph_visual(df_dict[img_idx],'subject','object','predicate',width=900,height=800)

    else:
        pass

        
        # 1) 이미지 번호로 이미지 추출




def Practice2():


    ttl_txt1_1 = "☑ SPO를 이용한 이미지 및 해당 이미지에 대한 Scene Graph 추출"
    st.markdown(f"""## <h1 class ="center">{ttl_txt1_1}</h1>""", unsafe_allow_html=True)
    st.text("")
    st.write("이미지에 검색할 **주어, 술어, 목적어**(**SPO**)를 입력해주세요.")

    sub2img_msg = "이미지를 검색할 **주어**(**Subject**)를 입력해주세요."
    pred2img_msg = "이미지를 검색할 **술어**(**Predicate**)를 입력해주세요."
    obj2img_msg = "이미지를 검색할 **목적어**(**Object**)를 입력해주세요."


    # 기본적으로 전체 selectbox가 빈값이 될수는 없어서 첫번째 selectbox만 'man'으로 지정해주기
    ## 추후 다른 값들은 디폴트가 아니라 selectbox list의 첫번째 요소로 넣어서 바로 나오게하기
    ### n번째 selectbox에 따라 가능한 항목 list들만 나오게함
    sub_list = list(tbl_scene['subject'].values)  
    default_ix1 = sub_list.index('man')
    input2 = ""
    input3 = ""

    col1 , col2, col3 = st.columns(3)
    with col1:
        input1 = st.selectbox(label = sub2img_msg, options = sub_list,
                              key = 1, disabled = False, index = default_ix1)
        input1.lower() # 소문자로 전부 통일


    with col2:
        if input1 != "":
            pred_list = list(tbl_scene.loc[tbl_scene.subject == input1]['predicate'].values)
            pred_list.insert(0, 'playing')
            input2 = st.selectbox(label = pred2img_msg, options = pred_list,
                              key = 2, disabled = False)
            input2.lower() # 소문자로 전부 통일
        #st.write(pred_list)


    with col3:
        if input1 != "" and input2 != "" :
            obj_list = list(tbl_scene.loc[(tbl_scene.subject == input1)&(tbl_scene.predicate == input2)]['object'].values)
            obj_list.insert(0, 'soccer')
            
            input3 = st.selectbox(label = obj2img_msg, options = obj_list,
                                  key = 3, disabled = False)#, index = default_ix3)
            input3.lower() # 소문자로 전부 통일
    col01,col02,col03=st.columns(3)
  

    if input1 != "" and input2 != "" and input3 != "":

        img_dic = get_spo(input1, input2, input3) #이미지와 해당 SPO테이블이 dictionary 형태로 저장되는 함수
        img_number = len(img_dic.keys())
        with col02:
            img_number_txt1 = f"<strong><i>{img_number}</i></strong> - Image is detected ❗"
            st.markdown(f"""<span style='text-align: left; color: #3b3b3b; font-size:120%'>{img_number_txt1}</span>""", unsafe_allow_html=True)
        st.markdown("___")
                    
        if img_number > 0:
    
            col1, col2, col3, col4 = st.columns([4.8, 0.2, 4.8, 0.2])

            if 'counter' not in st.session_state: 
                st.session_state.counter = 0

            # Get list of images in folder
            img_num_lst =  list(img_dic.keys())
            img_path = """/home/agens/conda_user/scene/aivg/data/action_genone/IMG_Action_Genome/VG_100K/"""
            filteredImages = [img_path + f"{img_num}.jpg" for img_num in img_num_lst]
            
            #filteredImages = [image_resize(image) for image in filteredImages]
            def showPhoto(photo,df):
                ## Increments the counter to get next photo
                st.session_state.counter += 1
                if st.session_state.counter >= len(filteredImages):
                    st.session_state.counter = 0

                with col03:
                    img_number_txt2 = f"<strong style='color: Black; font-size:150%'><i>{st.session_state.counter + 1}</i></strong>(th) out of {img_number}"
                    st.markdown(f"""###### {img_number_txt2}""", unsafe_allow_html=True)

                with col1:
                    res_ttl1 = f"Result 1 : <strong style = 'font-size : 120%'><i>Image</i></strong> matched by SPO"
                    st.markdown(f"""##### {res_ttl1}""",
                                unsafe_allow_html=True)
                    st.image(photo)
                with col3:
                    res_ttl2 = f"Result 2 : <strong style = 'font-size : 120%'><i>Scene Graph</i></strong> of Image matched by SPO"
                    st.markdown(f"""##### {res_ttl2}""",
                                unsafe_allow_html=True)
                    graph_visual(df, 'subject','object','predicate')



            

            # Select photo a send it to button
            photo = filteredImages[st.session_state.counter%img_number]
            df_idx = img_num_lst[st.session_state.counter%img_number]
            show_btn = col01.button("이미지 검색 결과 확인하기(계속)⏭️",on_click = showPhoto, args = ([photo, img_dic[df_idx]]))
    
    if input1 == "":
        st.write("❗ 주어를 입력(선택)해주세요")
    if input2 == "":
        st.write("❗ 술어를 입력(선택)해주세요")
    if input3 == "":
        st.write("❗ 목적어를 입력(선택)해주세요")
    if input1 == "" and input2 == "" and input3 == "":
        st.write("❗ **주어** 혹은 **술어** 혹은 **목적어**를 ***전부*** 입력해주세요.")
 
        
        


        
selected_menu = option_menu(
    None, 
    ["Intro", "Explanation", "Practice1 (IMG to SPO)", "Practice2 (SPO to IMG)"], 
    icons = ['bookmark-check', 'file-play-fill'], 
    menu_icon = "cast", 
    default_index = 0, 
    orientation = "horizontal",
    styles = {"container": {"padding": "5!important", "background-color": "#fafafa"},
    "icon": {"font-size": "21px"},
    "nav-link": {"font-size": "13.0px", "text-align": "center", "margin":"0px", "--hover-color": "#eee"},
    "nav-link-selected": {"background-color": "#007AFF",}})


    
if selected_menu == "Intro":
    Intro()
elif selected_menu == 'Explanation':
    Explanation()
elif selected_menu == "Practice1 (IMG to SPO)":
    Practice1()
elif selected_menu == "Practice2 (SPO to IMG)":
    Practice2()
    
