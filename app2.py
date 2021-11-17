import streamlit as st
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS,ImageColorGenerator
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import nltk
from nltk.corpus import stopwords
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
nltk.download('stopwords')
import numpy as np
from PIL import Image
def app():
    st.sidebar.header("Select Visualisation")
    visualisation = st.sidebar.selectbox('Visualisation',('Word cloud', 'Bar chart', 'count plot','pie chart','Hist plot','scatter plot','box plot','HeatMap' ))
    fake_job_postings_US= pd.read_csv(r'C:\Users\shrav\Downloads\fake_job_postings_us.csv')
    fake_job_postings_US=fake_job_postings_US.drop(columns=['Unnamed: 0'])

    def add_parameter_ui(clf_name):
        
        params = dict()
        if clf_name == 'Word cloud':
            st.sidebar.subheader("Feature")
            visual=st.sidebar.selectbox('Feature',('title','description','benefits'))
            params['visual'] = visual
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
    
    def sns_countplot(feature,rotation=0):
        fig, ax = plt.subplots(figsize=(10,5))
        ax=sns.countplot(x=feature, data=fake_job_postings_US, hue="fraudulent",
              order=fake_job_postings_US[feature].value_counts().iloc[:10].index)
        for p in ax.patches:
              ax.annotate(p.get_height(), (p.get_x() + p.get_width() / 2.0, p.get_height()),ha = 'center', va = 'center', xytext = (0, 5),textcoords = 'offset points')
        plt.xticks(rotation=rotation)
        title = feature + ' fake job count'
        plt.title(title)
        st.write(fig)
    def missing_count(feature,color):
        fig, ax = plt.subplots(figsize=(10,5))
        y_axis = fake_job_postings_US[fake_job_postings_US[feature].isna()][['fraudulent', feature]]
        y_axis = y_axis.fraudulent.value_counts()
        y_axis.plot(kind='bar',color=color)
        plt.ylabel('Count')
        plt.xlabel('Category')
        title = "Number of empty " + feature + " in fraudulent and non-fraudulent"
        plt.title(title)
        plt.xticks(rotation=0)
        st.write(fig)
    def word_cloud(feature,params):
        if params['image'] == 'default':
            fig, ax = plt.subplots()
            text = " ".join(feature for feature in fake_job_postings_US[feature].apply(str))
            stopwords=set(STOPWORDS)
            wordcloud = WordCloud(background_color="black",max_font_size=params['max_font_size'], max_words=params['max_words'],width=params['width'], height=params['height'],stopwords=stopwords,min_font_size=params['min_font_size'],min_word_length=params['min_word_length'],colormap=params['colormap']).generate(text)
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
            text = " ".join(feature for feature in fake_job_postings_US[feature].apply(str))
            stopwords=set(STOPWORDS)
            wordcloud = WordCloud(background_color="white",max_font_size=params['max_font_size'], max_words=params['max_words'],width=params['width'], height=params['height'],stopwords=stopwords,min_font_size=params['min_font_size'],min_word_length=params['min_word_length'],colormap=params['colormap'],mask = mask,contour_width=1, contour_color='firebrick').generate(text)
            image_colors = ImageColorGenerator(mask)
            wordcloud.recolor(color_func=image_colors)
            plt.imshow(wordcloud,interpolation="bilinear")
            plt.axis("off")
            plt.show()
            st.write(fig)                   
        
        
        
    def hist_plot(visual):
        fig,(ax1,ax2)= plt.subplots(ncols=2, figsize=(20, 10), dpi=100)
        num=fake_job_postings_US[fake_job_postings_US["fraudulent"]==1][visual].apply(str).str.split().map(lambda x: len(x))
        ax1.hist(num,bins = 20,color='orangered')
        ax1.set_title('Fake Post')
        num=fake_job_postings_US[fake_job_postings_US["fraudulent"]==0][visual].apply(str).str.split().map(lambda x: len(x))
        ax2.hist(num, bins = 20)
        ax2.set_title('Real Post')
        fig.suptitle('Words in {}'.format (visual))
        plt.show()
        st.write(fig)
        if visual == 'company_profile':
            st.header('The word pattern in the company profile is the same as the character pattern in the company profile. In the company profile of a fake post, there are less words than in a real post.')
        elif visual == 'description' :
            st.header('Both posts have a similar word distribution in their descriptions.')
        elif visual == 'requirements':
            st.header('The distribution of words in the fake and real post requirements is identical.')
     
    def hist(visual):
        fig,(ax1,ax2)= plt.subplots(ncols=2, figsize=(20, 10), dpi=100)
        length=fake_job_postings_US[fake_job_postings_US["fraudulent"]==1][visual].apply(str).str.len()
        ax1.hist(length,bins = 20,color='orangered')
        ax1.set_title('Fake Post')
        length=fake_job_postings_US[fake_job_postings_US["fraudulent"]==0][visual].apply(str).str.len()
        ax2.hist(length, bins = 20)
        ax2.set_title('Real Post')
        fig.suptitle('Characters in {}'.format (visual))
        plt.show()
        st.write(fig)
        if visual == 'company_profile':
            st.header('In the company profile, we can see that the fake message has fewer characters than the real post.')
        elif visual == 'description' :
            st.header('The distribution of characters in fake and real post descriptions is similar, but some fake posts have 6000 to 6500 characters.')
        elif visual == 'requirements':
            st.header('The distribution of charaters in requirements of the fake and real post are similar.')
        elif visual == 'benefits':
            st.header('Around 1500 to 1800, the proportion of characters in fake and real post benefits is the same.')
    def get_visualisation(visualisation):
        if visualisation == 'HeatMap':
            st.markdown("<h1 style='text-align: center; ';>HeatMap</h1>", unsafe_allow_html=True)  
            fig, ax = plt.subplots()
            sns.heatmap(fake_job_postings_US.corr(), ax=ax,annot = True,vmin=0, vmax=1,linewidths=.5,cmap="YlGnBu")
            st.write(fig)
        elif  visualisation == 'count plot':
            visual=st.sidebar.selectbox('Feature',('state_city','state','employment_type','required_experience','required_education','has_company_logo','telecommuting','fraudulent'))
            if visual == 'fraudulent':
                fig, ax = plt.subplots()
                sns.countplot(x='fraudulent', data=fake_job_postings_US,palette="dark")
                for p in ax.patches:
                    ax.annotate(p.get_height(), (p.get_x() + p.get_width() / 2.0, p.get_height()),ha = 'center', va = 'center', xytext = (0, 5),textcoords = 'offset points')
                st.write(fig)
            else:
                if visual== 'required_education' :
                    sns_countplot(visual,90)
                else:
                    sns_countplot(visual,45)

        elif visualisation == 'pie chart':
            a=fake_job_postings_US[fake_job_postings_US.fraudulent==0]['employment_type'].value_counts()
            a.values
            b=fake_job_postings_US[fake_job_postings_US.fraudulent==1]['employment_type'].value_counts()
            b.values

            labels = ['Full-time', 'Contract', 'Part-time', 'Other', 'Temporary']
            st.subheader("Employment_type vs Fraudulent")
            fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
            fig.add_trace(go.Pie(labels=labels, values=a.values, name="Real"),1, 1)
            fig.add_trace(go.Pie(labels=labels, values=b.values, name="Fake"), 1, 2)
            fig.update_traces(hole=.4, hoverinfo="label+percent+name")
            fig.update_layout(annotations=[dict(text='Real', x=0.18, y=0.5, font_size=20, showarrow=False),dict(text='Fake', x=0.82, y=0.5, font_size=20, showarrow=False)])
            st.write(fig)
        elif visualisation == 'Hist plot':
            visual = st.sidebar.selectbox('visualize',('Frequency of Words', 'Number of characters','Number of words'))
            if visual == 'Frequency of Words':
                fake_job_postings_US.fillna(" ",inplace = True)

                fake_job_postings_US['text'] =  fake_job_postings_US['title'] + ' ' + fake_job_postings_US['location'] + ' ' + fake_job_postings_US['company_profile'] + ' ' + \
                fake_job_postings_US['description'] + ' ' + fake_job_postings_US['requirements'] + ' ' + fake_job_postings_US['benefits'] + ' ' + \
                fake_job_postings_US['required_experience'] + ' ' + fake_job_postings_US['required_education'] + ' ' + fake_job_postings_US['industry'] + ' ' + fake_job_postings_US['function']

                fake_job_postings_US['character_count'] = fake_job_postings_US.text.apply(len)
                fig, ax = plt.subplots()
                fake_job_postings_US[fake_job_postings_US.fraudulent==0].character_count.plot(bins=35, kind='hist', color='yellow', 
                                           label='Real', alpha=0.8,edgecolor='red')
                fake_job_postings_US[fake_job_postings_US.fraudulent==1].character_count.plot(kind='hist', color='green', 
                                           label='Fake', alpha=0.8,edgecolor='blue')

                plt.legend()
                plt.title('Frequency of Words')
                plt.xlabel("Character Count");
                st.write(fig)
            elif visual == 'Number of words':
                visual = st.sidebar.radio('',('description','company_profile','requirements','benefits'))
                hist_plot(visual)
            elif visual == 'Number of characters':
                visual = st.sidebar.radio('',('description','company_profile','requirements','benefits'))
                hist(visual)
                
                

        elif visualisation == 'Word cloud': 
            
            word_cloud(params['visual'],params)
        elif visualisation == 'scatter plot':
            a=fake_job_postings_US[fake_job_postings_US.fraudulent==0]['telecommuting'].value_counts()
            b=fake_job_postings_US[fake_job_postings_US.fraudulent==1]['telecommuting'].value_counts()
            df=pd.DataFrame({'type':['Real','Fake','Real','Fake'],'telicom':['work_from_office','work_from_office','work_from_home','work_from_home'],'count':[a[0],b[0],a[1],b[1]]})
            fig = px.scatter(df, y="type", x="count",color="telicom", symbol="telicom")
            fig.update_traces(marker_size=10)
            st.plotly_chart(fig)
        elif visualisation == 'box plot':
            fake_job_postings_US.fillna(" ",inplace = True)

            fake_job_postings_US['text'] =  fake_job_postings_US['title'] + ' ' + fake_job_postings_US['location'] + ' ' + fake_job_postings_US['company_profile'] + ' ' + \
                fake_job_postings_US['description'] + ' ' + fake_job_postings_US['requirements'] + ' ' + fake_job_postings_US['benefits'] + ' ' + \
            fake_job_postings_US['required_experience'] + ' ' + fake_job_postings_US['required_education'] + ' ' + fake_job_postings_US['industry'] + ' ' + fake_job_postings_US['function']

            fake_job_postings_US['character_count'] = fake_job_postings_US.text.apply(len)
            fig = px.box(fake_job_postings_US, x="fraudulent", y="character_count")
            st.plotly_chart(fig)  
        elif visualisation == 'Bar chart':
            visual=st.sidebar.selectbox('Feature',('function', 'company_profile','required_education','industry','benefits','state'))
            color = st.sidebar.color_picker('Pick A Color', '#00f900')
            if visual =='state':
                fig, ax = plt.subplots(figsize=(10,5))
                fake_job_postings_US.groupby('state').fraudulent.count().plot(kind='bar', title='Job count by states',color=color)
                st.write(fig)
            else:
                missing_count(visual,color)
            
    get_visualisation(visualisation)
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    