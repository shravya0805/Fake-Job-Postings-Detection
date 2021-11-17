import streamlit as st
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS,ImageColorGenerator
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from PIL import Image
def app():
    st.sidebar.header("Select Visualisation")
    visualisation = st.sidebar.selectbox('Visualisation',('Word cloud','HeatMap', 'Bar chart', 'count plot','pie chart','Hist plot','scatter plot','box plot' ))
    Reviews = pd.read_csv(r'C:\Users\shrav\Downloads\Reviews.csv')
    def add_parameter_ui(clf_name):
        
        params = dict()
        if clf_name == 'Word cloud':

            st.sidebar.subheader("Hyperparameters")
            color=list({'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'crest', 'crest_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'flare', 'flare_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'icefire', 'icefire_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'viridis'})
            colormap = st.sidebar.text_input("enter color", 'viridis')
            if colormap in color:
                params['colormap'] = colormap
            else:
                st.subheader('invalid color.pls enter the following colors from the below')
                st.write('{}'.format(color))
            image = st.sidebar.selectbox(label='Select Image Mask',options=['default','elephant','wine','India','crab','twitter','Trump','geeksforgeeks'])
            params['image']=image
            width = st.sidebar.slider("width", 400, 1000,  key="width")
            params['width'] = width 
            height=st.sidebar.slider("height", 200, 1000,  key="height")
            params['height']=height
            min_font_size=st.sidebar.slider("min_font_size", 4, 10,  key="min_font_size")
            params['min_font_size'] = min_font_size
            max_words=st.sidebar.slider("max_words", 200, 1000,  key="max_words")
            params['max_words'] = max_words
            max_font_size=st.sidebar.slider("max_font_size", 100, 200,  key="max_font_size")
            params['max_font_size'] = max_font_size
            min_word_length=st.sidebar.slider("min_word_length", 0, 50,  key="min_word_length")
            params['min_word_length'] = min_word_length
            
        return params
    params = add_parameter_ui(visualisation)
    
    def word_cloud(feature,params):
        data_recommended =Reviews[Reviews['Recommended IND'] == 1]  
        data_not_recommended = Reviews[Reviews['Recommended IND'] == 0] 
        if params['image'] == 'default':
            fig, ax = plt.subplots()
            if feature == 'data_not_recommended':
                wordcloud = WordCloud(background_color="black",max_font_size=params['max_font_size'], max_words=params['max_words'],width=params['width'], height=params['height'],min_font_size=params['min_font_size'],min_word_length=params['min_word_length'],colormap=params['colormap']).generate(str((data_not_recommended['Review Text'])))
                plt.imshow(wordcloud,interpolation="bilinear")
                plt.axis("off")
                plt.show()
                st.write(fig)   
            else :
                wordcloud = WordCloud(background_color="black",max_font_size=params['max_font_size'], max_words=params['max_words'],width=params['width'], height=params['height'],min_font_size=params['min_font_size'],min_word_length=params['min_word_length'],colormap=params['colormap']).generate(str((data_recommended['Review Text'])))
                plt.imshow(wordcloud,interpolation="bilinear")
                plt.axis("off")
                plt.show()
                st.write(fig)   
        else:
            image=params['image'] 
            list =['Trump','India','crab','wine','geeksforgeeks']
            if image in list:
                path = r'C:\Users\shrav\Downloads\images\{}.png'.format(image)
            else:
                path = r'C:\Users\shrav\Downloads\images\{}.jpg'.format(image)
            mask = np.array(Image.open(path))
            fig, ax = plt.subplots()
            if feature == 'data_not_recommended':
                wordcloud = WordCloud(background_color="white",max_font_size=params['max_font_size'], max_words=params['max_words'],width=params['width'], height=params['height'],min_font_size=params['min_font_size'],min_word_length=params['min_word_length'],colormap=params['colormap'],mask = mask,contour_width=1, contour_color='firebrick').generate(str((data_not_recommended['Review Text'])))
                image_colors = ImageColorGenerator(mask)
                wordcloud.recolor(color_func=image_colors)
                plt.imshow(wordcloud,interpolation="bilinear")
                plt.axis("off")
                plt.show()
                st.write(fig)   
            else :
                wordcloud = WordCloud(background_color="white",max_font_size=params['max_font_size'], max_words=params['max_words'],width=params['width'], height=params['height'],min_font_size=params['min_font_size'],min_word_length=params['min_word_length'],colormap=params['colormap'],mask = mask,contour_width=1, contour_color='firebrick').generate(str((data_recommended['Review Text'])))
                image_colors = ImageColorGenerator(mask)
                wordcloud.recolor(color_func=image_colors)
                plt.imshow(wordcloud,interpolation="bilinear")
                plt.axis("off")
                plt.show()
                st.write(fig)   
                
    
    def get_visualisation(visualisation):
        if visualisation == 'HeatMap':
            fig, ax = plt.subplots()
            sns.heatmap(Reviews.corr(), ax=ax,annot = True)
            st.write(fig)
        elif  visualisation == 'count plot':
            visual=st.sidebar.radio('count plot',['Distribution of ratings', 'rating and recommendation','deparment vs rating','Department Name vs Recommended IND','Division Name vs Recommended IND'])
            if visual == 'Distribution of ratings':
                fig, ax = plt.subplots()
                sns.countplot(data = Reviews , x = 'Rating',ax=ax)
                st.write(fig)
            elif visual =='rating and recommendation':
                fig, ax = plt.subplots()
                sns.countplot(x=Reviews['Rating'],hue = Reviews['Recommended IND'])
                st.write(fig)
            elif visual == 'deparment vs rating':
                fig, ax = plt.subplots()
                sns.countplot(x=Reviews['Rating'],hue = Reviews['Department Name'])
                st.write(fig)
            elif visual == 'Department Name vs Recommended IND':
                fig, ax = plt.subplots()
                sns.countplot(x=Reviews['Department Name'],hue = Reviews['Recommended IND'])
                st.write(fig)
            elif visual == 'Division Name vs Recommended IND':
                fig, ax = plt.subplots()
                sns.countplot(x=Reviews['Division Name'],hue = Reviews['Recommended IND'])
                st.write(fig)
        elif visualisation == 'pie chart':
            labels = ['Recommended', 'Not Recommended']
            values = [Reviews[Reviews['Recommended IND'] == 1]['Recommended IND'].value_counts()[1],Reviews[Reviews['Recommended IND'] == 0]['Recommended IND'].value_counts()[0]]
            fig, ax = plt.subplots()
            plt.pie(values , labels = labels)
            st.write(fig)
        elif visualisation == 'Hist plot':
            visual=st.sidebar.radio('wordcloud',['length_of_text', 'type of dress vs count'])
            if visual =='length_of_text':
                Reviews['length_of_text'] = [len(i.split(' ')) for i in Reviews['Review Text']]
                fig, ax = plt.subplots()
                sns.histplot(Reviews['length_of_text'])
                st.write(fig)
            else:
                fig=px.histogram(Reviews, x= 'Class Name')
                st.plotly_chart(fig)  
        elif visualisation == 'Word cloud':
            visual=st.sidebar.radio('wordcloud',['WordCloud of the Recommended Reviews', 'WordCloud of the Not_Recommended Reviews'])
            if visual=='WordCloud of the Recommended Reviews':    
                word_cloud('data_recommended',params)
            else :
                word_cloud('data_not_recommended',params)
        elif visualisation == 'scatter plot':
            fig=px.scatter(Reviews, x="Age", y="Positive Feedback Count")
            st.plotly_chart(fig)
        elif visualisation == 'box plot':
            fig=px.box(Reviews, x="Age", y="Division Name", orientation="h",color = 'Recommended IND')
            st.plotly_chart(fig)    
    get_visualisation(visualisation)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    