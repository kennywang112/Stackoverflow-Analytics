import streamlit as st 
import altair as alt 
import requests
import string
import re
import matplotlib.pyplot as plt 
import pandas as pd 
from wordcloud import WordCloud,STOPWORDS
from nltk.corpus import stopwords
from bs4 import BeautifulSoup 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

def text_mining(page) :
    url = 'https://stackoverflow.com/questions?tab=newest&page=' + str(1)
    res = requests.get(url)
    soup = BeautifulSoup(res.text,'html.parser')
    post_titles = soup.find_all('a', {'class' : "s-link"})
    hot_network_question = soup.find_all('a', {'href' : "https://stackexchange.com/questions?tab=hot"})
    tags_in_posts = soup.find_all('ul', {'class' : "ml0 list-ls-none js-post-tag-list-wrapper d-inline"})

    ml = {}
    specific = []
    l = 0
    for tags_in_post in tags_in_posts :
        if 'java' in tags_in_post.text:
            specific.append(1)
        else:
            specific.append(0)
    # confirm it's not belongs to the row 'Hot Network Questions'
    for title in post_titles[len(post_titles) - len(tags_in_posts) - 1 : -1] :
        if hot_network_question[0].text != title.text :
            ml[title.text] = specific[l]
            l += 1
    return ml
# remove specific characters, drop numbers, remove stopwords
def remove_special_characters(text):
    pat = r'[^a-zA-z0-9]' 
    return re.sub(pat, ' ', text)

def drop_numbers(list_text):
    list_text_new = []
    for i in list_text:
        if not re.search('\d', i ):
            list_text_new.append(i)
    return ''.join(list_text_new)

def remove_stopwords(text) :
    text = [word.lower() for word in text.split() if word.lower() not in stop]
    return " ".join(text)

def company(page) :
    url_companies = 'https://stackoverflow.com/jobs/companies?pg=' + str(page)
    res_comp = requests.get(url_companies)
    soup_comp = BeautifulSoup(res_comp.text, "html.parser")
    company_name = soup_comp.find_all('a', {'class' : "s-link"})
    company_loc_indus = soup_comp.find_all('div', {'class' : "flex--item fc-black-500 fs-body1"})
    company_tag = soup_comp.find_all('a', {'class' : "flex--item s-tag no-tag-menu"})
    page_limit_company = soup_comp.find_all('div', {'class' : "dismissable-company -company ps-relative js-dismiss-overlay-container p24 pr32 bb bc-black-100"})
    company_name_lst, company_tag_lst, comp_location_industry, comp_location, comp_industry = [], [], [], [], []
    dic_company = {}
    #process company name
    for name in company_name[2: ] :
        company_name_lst.append(name.text)
    #process company text
    for tags in company_tag :
        company_tag_lst.append(tags.text)
    #process company industry and location
    for i in company_loc_indus :
        comp_location_industry.append(i.text)
    for i in range(len(company_loc_indus)):
        #comp_location.append(comp_location_industry[i]) if i%2==0 else comp_industry.append(comp_location_industry[i])
        if i % 2 == 0:
            comp_location.append(comp_location_industry[i])
        else:
            comp_industry.append(comp_location_industry[i])
    #process each pages data  
    for m in range(len(page_limit_company)):
        dic_company[company_name_lst[m]] = company_tag_lst[0+3*m : 3+3*m]
        
    return dic_company, company_tag_lst, comp_location, comp_industry

def to_df() :
    #turn array and dictionary into dataframe
    company_tag_freq = {}
    #process company,tags,location,industry columns
    df_company_and_tags = pd.DataFrame(columns = ['company', 'tags'])
    df_loc_indus = pd.DataFrame()
    for i in range(1, 16) :
        #write all tag into dic , and get the frequency of tag
        comp = company(i)
        for tag in comp[1] :
            company_tag_freq[tag] = 1 if tag not in company_tag_freq else company_tag_freq[tag] + 1
        #each companys' tags
        company_and_tags = comp[0]
        cat = pd.DataFrame(company_and_tags.items(), columns = ['company', 'tags'])
        lai = pd.DataFrame(comp[2:]).T
        df_company_and_tags = pd.concat([df_company_and_tags,cat])
        df_loc_indus = pd.concat([df_loc_indus,lai])
    data = pd.concat([df_company_and_tags,df_loc_indus],axis = 1)
    data.columns = ['company', 'tags', 'location', 'industry']
    
    return data, company_tag_freq

def clean(doc) :
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punct_free = ''.join(ch for ch in stop_free if ch not in punct_exclude)
    num_free = ''.join(i for i in punct_free if not i.isdigit())
    return num_free

url = 'https://stackoverflow.com/tags?page=1&tab=popular'
res = requests.get(url)
soup = BeautifulSoup(res.text,'html.parser')
posts_infos = soup.find_all('div', {'class':"mt-auto d-flex jc-space-between fs-caption fc-black-400"})
tags = soup.find_all('a',{'class':'post-tag'})

post_lst = []
tags_lst = []
dic_ask = {'tag':[],'question':[],'asked today':[],'question this week':[]}

for post_infos in posts_infos:
    post_lst.append(post_infos.text)   
for tag in tags:
    tags_lst.append(tag.text)
# replace useless strings
for j in range(len(post_lst)) :
    post_lst[j] = post_lst[j].replace('\n','')
    post_lst[j] = post_lst[j].replace(' questions ',',')
    post_lst[j] = post_lst[j].replace(' asked today, ',',')
    post_lst[j] = post_lst[j].replace(' this week ','')
    post_lst[j] = post_lst[j].replace(' asked this week','')
    post_lst[j] = post_lst[j].replace(' this month ','')
    post_lst[j] = post_lst[j].split(',')   
# put each number into different key (ask this week,ask today and total ask)
for i in post_lst :
    dic_ask['question'].append(i[0])
    dic_ask['asked today'].append(i[1])
    dic_ask['question this week'].append(i[2])   
    
for i in tags:
    dic_ask['tag'].append(i.text)
    
dic_ask['question'] = pd.to_numeric(dic_ask['question'])
dic_ask['asked today'] = pd.to_numeric(dic_ask['asked today'])
dic_ask['question this week'] = pd.to_numeric(dic_ask['question this week'])
df_ask = pd.DataFrame(dic_ask,columns = ['tag','question', 'asked today', 'question this week'])
df_ask = df_ask.sort_values('question this week', ascending = False)

big_title = '<p style="font-family:sans-serif; color:Green; font-size: 42px;">StackOverflow Data</p>'
st.markdown(big_title, unsafe_allow_html=True)
expander_bar = st.expander('About')
expander_bar.markdown("""
**data source :** [stackoverflow](https://stackoverflow.com/)""")

chart = alt.Chart(df_ask).mark_circle().encode(
    x = 'asked today',
    y = 'question this week',
    color = 'tag',
).interactive()
chart2 = alt.Chart(df_ask).mark_circle().encode(
    x = 'asked today',
    y = 'question',
    color = 'tag',
).interactive()

st.sidebar.header('User Input Features')
selectcolumn = st.sidebar.selectbox('ASK of every tags today/this week/total', ['question', 'asked today', 'question this week'])
st.subheader('dataframe & chart')
p = alt.Chart(df_ask).mark_bar().encode(
    x = alt.X('tag', sort = None),
    y = selectcolumn,
)
p = p.properties(
    width = alt.Step(20)
)

col1, col2 = st.columns((2,1))
cold, cold2 = st.columns((1,1))
col1.write(df_ask)
col2.write(p)
cold.write(chart)
cold2.write(chart2)

predict_title = '<p style="font-family:sans-serif; color:Green; font-size: 42px;">Text Classification Using sklearn</p>'
st.markdown(predict_title, unsafe_allow_html = True)
#st.title('Text Classification Using sklearn')
st.subheader('Is the question about java ? (yes : 1 , no : 0) & Java tag bar chart')

slide_page = st.sidebar.slider('page of questions for machine learning ', 2, 10, 6)
data_total = []
for slide in range(1, slide_page) :
    text_data = text_mining(slide)
    data_total.append(text_data)
    
data_ml = pd.concat(pd.DataFrame(data_total[i].items()) for i in range(len(data_total)))
data_ml.columns = ['text', 'java_tag']
# process data_ml, keep a copy 
data_copy = data_ml
# tag count
tag_count = pd.DataFrame(data_ml['java_tag'].value_counts())
# st.write(tag_count)
tag_count.index.name = 'tag'
tag_count.reset_index(inplace = True)

col3, col4 = st.columns((5, 1))
col3.write(data_ml.head(30))
with col4:
    st.bar_chart(data = tag_count, x = 'tag', y = 'count', use_container_width = True)

# filter text
data_ml['text'] = data_ml.apply(lambda x: remove_special_characters(x['text']), axis = 1)
data_ml['text'] = data_ml['text'].apply(drop_numbers)
stop = set(stopwords.words("english"))
data_ml["text"] =data_ml["text"].map(remove_stopwords)

# creating bag of words
cv = CountVectorizer(max_features = 2500)

x = cv.fit_transform(data_ml['text']).toarray()
y = data_ml['java_tag'].values
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0,shuffle = False)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
pred = classifier.predict(X_test)

#process the predict sentences
train_test_target = y_train.tolist()+pred.tolist()
train_test_target = pd.DataFrame(train_test_target)
data_copy.reset_index(drop = True, inplace = True)
train_test_target.reset_index(drop=True, inplace = True)
new_data_ml = pd.concat([data_copy,train_test_target],axis = 1)
pred_sentence = []

for i in range(round(0.75 * len(new_data_ml)) - 1, len(new_data_ml)) :
    if new_data_ml[0][i] == 1 :
       pred_sentence.append(new_data_ml['text'][i])
    
frame_pred = pd.DataFrame(pred_sentence)
frame_pred.columns = ['Predict Sentences']

st.subheader('Data training and test')
st.write('training score :', classifier.score(X_train, y_train),'test score :', classifier.score(X_test, y_test))
st.write("The test score is: ", round(accuracy_score(y_test, pred) * 100,2), '%')
acc_dic = classification_report(y_test, pred, output_dict = True)
acc_df = pd.DataFrame(acc_dic).transpose()
st.write(acc_df)
st.subheader('Sentences predicted have a java tag :sunglasses:')
st.write(frame_pred)
st.subheader("Companys' news")

company_data = to_df()
#plot each tags' frequency
company_tag = pd.DataFrame(company_data[1].items(),columns = ['tagged', 'freq'])
company_tag = company_tag.sort_values('freq',ascending = False)
new_company_tag = company_tag[company_tag.freq > 2]
comp_plot = alt.Chart(new_company_tag).mark_bar().encode(
    x = alt.X('tagged', sort = None),
    y = 'freq',
)
comp_plot = comp_plot.properties(
    width = alt.Step(20)
)
st.write(comp_plot) 
st.write(company_data[0])
st.subheader('Most count words in new questions')

m = 0
dic = {}
for i in range(10) :
    url = 'https://stackoverflow.com/questions?tab=newest&page=' + str(i)
    res = requests.get(url)
    soup = BeautifulSoup(res.text,'html.parser')
    posts_title = soup.find_all('a', {'class':"s-link"})
    for title in posts_title[2:] :
        dic[m] = title.text
        m += 1
        
df = pd.DataFrame(dic.items())
stop = set(stopwords.words())
punct_exclude = set(string.punctuation)

post_corpus = [clean(df.iloc[i, 1]) for i in range(0, df.shape[0])]
wordcloud = WordCloud(width = 1000, height = 500, stopwords = STOPWORDS, background_color = 'white', font_path = 'RadiantKingdom-mL5eV.ttf').generate(''.join(post_corpus))
plt.axis('off')
plt.tight_layout()
plt.imshow(wordcloud)
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()