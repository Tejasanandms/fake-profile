import os
import pytesseract
import cv2
#-------------------------------------------------model_code------------------------------------------------------------
import sqlite3
conn = sqlite3.connect('fake_account')
cur = conn.cursor()
try:
   cur.execute('''CREATE TABLE user (
     name varchar(20) DEFAULT NULL,
      email varchar(50) DEFAULT NULL,
     password varchar(20) DEFAULT NULL,
     gender varchar(10) DEFAULT NULL,
     age int(11) DEFAULT NULL
   )''')

except:
   pass
#!/usr/bin/env python
# coding: utf-8



#include packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.naive_bayes import MultinomialNB

data_Fake = pd.read_csv("fusers.csv")
data_User = pd.read_csv("users.csv")

data_Fake = data_Fake.drop(['id','name','screen_name','protected','created_at','verified','default_profile_image','url','time_zone','profile_banner_url','geo_enabled','profile_background_tile','utc_offset'],axis=1)
data_Fake = data_Fake.dropna()
data_User = data_User.drop(['id','name','screen_name','protected','created_at','verified','default_profile_image','url','time_zone','profile_banner_url','geo_enabled','profile_background_tile','utc_offset'],axis=1)
data_User = data_User.dropna()
Data = pd.concat([data_Fake,data_User])
x = Data.iloc[:,:6]
y = Data.iloc[:,-1:]

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

lang = le.fit_transform(x.lang)
x['lang'] = lang

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.20,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#nb = GaussianNB()
nb = RandomForestClassifier()
nb.fit(X_train,Y_train)

Y_pred_nb = nb.predict(X_test)

score_nb = round(accuracy_score(Y_pred_nb,Y_test)*100,2)

print("The accuracy score achieved using random forest is: "+str(score_nb)+" %")


from flask import Flask,render_template, url_for,request, flash, redirect, session
app = Flask(__name__)
app.config['SECRET_KEY'] = '881e69e15e7a528830975467b9d87a98'
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
#-------------------------------------home_page-------------------------------------------------------------------------

@app.route('/')
@app.route('/home')
def home():
   if not session.get('logged_in'):
      return render_template('home.html')
   else:
      return redirect(url_for('user_account'))
#-------------------------------------about_page-------------------------------------------------------------------------
@app.route("/about")
def about():
   return render_template('about.html')
#-------------------------------------about_page-------------------------------------------------------------------------

#-------------------------------------user_login_page-------------------------------------------------------------------------
@app.route('/user_login',methods = ['POST', 'GET'])
def user_login():
   conn = sqlite3.connect('fake_account')
   cur = conn.cursor()
   if request.method == 'POST':
      email = request.form['email']
      password = request.form['psw']
      print('asd')
      count = cur.execute('SELECT * FROM user WHERE email = "%s" AND password = "%s"' % (email, password))
      print(count)
      #conn.commit()
      #cur.close()
      l = len(cur.fetchall())
      if l > 0:
         flash( f'Successfully Logged in' )
         return render_template('user_account.html')
      else:
         print('hello')
         flash( f'Invalid Email and Password!' )
   return render_template('user_login.html')

# -------------------------------------user_login_page-----------------------------------------------------------------

# -------------------------------------user_register_page-------------------------------------------------------------------------

@app.route('/user_register', methods=['POST', 'GET'])
def user_register():
    conn = sqlite3.connect('fake_account')
    cur = conn.cursor()
    if request.method == 'POST':
       name = request.form['uname']
       email = request.form['email']
       password = request.form['psw']
       gender = request.form['gender']
       age = request.form['age']
       cur.execute('SELECT * FROM user WHERE email = "%s"' % (email))
       count = cur.fetchall()
       if len(count) == 0 :
            cur.execute("insert into user(name,email,password,gender,age) values ('%s','%s','%s','%s','%s')" % (name, email, password, gender, age))
            conn.commit()
            print('data inserted')
            return redirect(url_for('user_login'))
       cur.close()
       flash('clone account')

    return render_template('user_register.html')
# -------------------------------------user_register_page-------------------------------------------------------------------------

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    statuses_count = request.form['statuses_count']
    followers_count = request.form['followers_count']
    friends_count = request.form['friends_count']
    favourites_count = request.form['favourites_count']
    listed_count = request.form['listed_count']
    lang = request.form['lang']
    global nb
    if request.method == 'POST':
        out = nb.predict([[float(statuses_count), float(followers_count), float(friends_count), float(favourites_count),
                           float(listed_count), float(lang)]])
        #print(out)
        if out[0] == 0 or float(lang)<10:
            print('No')
            flash(f'It is not a Fake Account')
        else:
            print('Yes')
            flash(f' It is a Fake Account')
        return render_template('user_account.html')

@app.route('/fakenews', methods=['POST', 'GET'])
def fakenews():
    return render_template('server.html')

@app.route('/search', methods=['POST', 'GET'])
def search():py
    return render_template('search.html')

@app.route('/analyse', methods=['POST', 'GET'])
def analyse():
    from sklearn.naive_bayes import MultinomialNB
    if request.method == 'POST':
        f1 = request.files['file1']
        print(r'{}'.format(f1.filename ))
        #img_cv = cv2.imread(r'IMG-20210701-WA0005.jpg')
        img_cv = cv2.imread(r'{}'.format(f1.filename ))
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract'
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        print(pytesseract.image_to_string(img_rgb))
        data = [pytesseract.image_to_string(img_rgb)]

        df = pd.read_csv("fake.csv", encoding="latin-1")
        df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
        df['label'] = df['v1'].map({'ham': 0, 'spam': 1})
        X = df['v2']
        y = df['label']
        cv = CountVectorizer()
        X = cv.fit_transform(X)  # Fit the Data
        #from sklearn.model_selection import train_test_split
        # from sklearn.cross_validation import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        # from sklearn.naive_bayes import MultinomialNB
        clf = MultinomialNB()
        clf.fit(X_train, y_train)
        clf.score(X_test, y_test)

        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)

        if 0 == my_prediction[0]:
            #e4.delete(0, END)
            #e4.insert(0, 'Not a Fake')
            flash(f'Not a fake')
        else:
            #e4.delete(0, END)
            #e4.insert(0, 'Fake')
            flash('Fake news')
    return render_template('search.html')


@app.route("/logout")
def logout():
   session['logged_in'] = False
   return home()

# ------------------------------------predict_page-----------------------------------------------------------------
if __name__ == '__main__':
   app.secret_key = os.urandom(12)
   app.run(debug=True)
