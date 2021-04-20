import streamlit as st 
import plotly.express as px
import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd 
import folium
from streamlit_folium import folium_static
import numpy as np 
import folium
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from keras import models
from keras import layers,optimizers,metrics,losses
import datetime
pd.set_option('display.max_columns', None)
from keras.optimizers import  Adagrad,Adam,Adadelta,SGD,RMSprop
from keras.layers import ReLU,LeakyReLU,ELU
import keras
import sklearn
from urllib.request import urlretrieve
from streamlit_folium import folium_static
from keras.utils.vis_utils import plot_model
import time
import io
from keras.models import load_model
import base64
import json


def main():

    html_temp = """
            <div style="background-color:royalblue;padding:10px;border-radius:10px">
            <h1 style="color:white;text-align:center;">Number of Vehicles Prediction  in Besiktas, Istanbul Using Neural Networks</h1>
            </div>
            """
    st.markdown(html_temp, unsafe_allow_html = True) 


    page_choice=st.sidebar.radio("Pages",["Prediction","Plots","Codes"])


    def create_space(number_of_row):
        for i in range(number_of_row):

            st.sidebar.markdown("&nbsp;")

    create_space(5)

    st.sidebar.markdown("**Info**\n \nWriter : Ferhat Metin \nferhatmetin34@gmail.com")


    st.sidebar.markdown(""" 
                            <a href="https://linkedin.com/in/ferhat-metin" target="blank"><img align="center" 
                            src="https://cdn.jsdelivr.net/npm/simple-icons@3.0.1/icons/linkedin.svg" alt="ferhat-metin" height="30" width="30" /></a><a
                            href="https://kaggle.com/ferhatmetin34" target="blank"><img align="center"
                            src="https://cdn.jsdelivr.net/npm/simple-icons@3.0.1/icons/kaggle.svg" alt="ferhatmetin34" height="30" width="30" /></a><a
                            href="https://github.com/ferhatmetin34" target="blank"><img align="center"
                            src="https://cdn.jsdelivr.net/npm/simple-icons@3.0.1/icons/github.svg" alt="ferhatmetin34" height="30" width="30" /></a></p>
                            
                        """, unsafe_allow_html = True)


    @st.cache(allow_output_mutation=True)
    def read_files(path):
        return pd.read_csv(path,index_col=0)

    @st.cache
    def convert_to_date(df):
        df["date_time"]=pd.to_datetime(df["date_time"])

    @st.cache
    def create_data(df,i):
        koordinatlar=list(set(zip(df.latitude,df.longitude)))
        long=koordinatlar[i][1]
        lat=koordinatlar[i][0]
        #print(long,lat)
        new_data=df[(df.latitude==lat)&(df.longitude==long)].sort_values(by="date_time")
        return new_data

    @st.cache
    def extract_date(data):
        data["hour"]=data.date_time.dt.hour
        data["month"]=data.date_time.dt.month
        data["week"]=data.date_time.dt.week
        data['dayofweek'] = data.date_time.dt.dayofweek
        data["dayofmonth"]=data.date_time.dt.day
        data["day_name"]=data.date_time.dt.day_name()
        data['is_weekend'] = np.where(data['day_name'].isin(['Sunday', 'Saturday']), 1,0)
        return data

    @st.cache
    def error_table(y_pred,y_test):
        mse=mean_squared_error(y_test,y_pred.ravel())
        rmse=np.sqrt(mse)
        mae=mean_absolute_error(y_test,y_pred.ravel())
        r2=r2_score(y_test,y_pred)
        mape=np.mean(np.abs((y_test - y_pred.ravel()) / y_test)) * 100
        table_error=pd.DataFrame({"MSE":mse,"RMSE":rmse,"MAE":mae,"R2":r2,"MAPE":mape},index=[0])
        return table_error

    @st.cache(allow_output_mutation=True)
    def create_map():
        loc_df=pd.DataFrame((

        (41.0421752929688, 29.0093994140625),

        (41.0476684570312,29.0093994140625),                   

        (41.0366821289062, 28.9984130859375),
    
        (41.0531616210938, 29.0093994140625),

        (41.0586547851562, 29.0093994140625),

        (41.0476684570312, 28.9874267578125),

        (41.0421752929688, 28.9874267578125),

        (41.0366821289062, 28.9874267578125),

        (41.0311889648438, 28.9874267578125),

        (41.0586547851562, 28.9984130859375),

        (41.0641479492188, 29.0093994140625),

        (41.0476684570312, 29.0203857421875),

        (41.0421752929688, 28.9984130859375),

        (41.0476684570312, 28.9984130859375),

        (41.0531616210938, 28.9984130859375),

        (41.0641479492188, 28.9984130859375)),  columns=["latitude","longitude"])

        loc_data=loc_df.copy()

        m = folium.Map([41,29], zoom_start=5,width="%100", height="%100")

        for index, loc_df in loc_df.iterrows():
                location = [loc_df['latitude'], loc_df['longitude']]
                folium.Marker(location, popup =f'{loc_df["latitude"],loc_df["longitude"]}' ).add_to(m)

        folium.TileLayer('openstreetmap').add_to(m)
        folium.TileLayer('stamentoner').add_to(m)
        folium.TileLayer("cartodbdark_matter").add_to(m)
        folium.TileLayer("stamenterrain").add_to(m)
        folium.TileLayer("stamenwatercolor").add_to(m)
        folium.LayerControl().add_to(m)

        return loc_data,m

    @st.cache
    def prepare_data(df,weather_cond):

        full_data=pd.merge(df,weather_cond,
                            on=["date_time"],
                            how="left").drop_duplicates("date_time").reset_index(drop=True)


        full_data=extract_date(full_data)

        columns=['date_time','minimum_speed', 'maximum_speed', 'average_speed', 'number_of_vehicles', 
                'WindGustKmph', 'cloudcover', 'humidity', 'maxtempC', 'mintempC',
                'precipMM', 'tempC',"hour","month","dayofweek","dayofmonth","is_weekend"]
        #'winddirDegree'
        full_data=full_data[columns]


        full_data["arac_sayi(t-1)"]=full_data.number_of_vehicles.shift(1)
        full_data["arac_sayi(t-2)"]=full_data.number_of_vehicles.shift(2)
        full_data["arac_sayi(t-3)"]=full_data.number_of_vehicles.shift(3)
        full_data["arac_sayi(t-4)"]=full_data.number_of_vehicles.shift(4)
        full_data["arac_sayi(t-5)"]=full_data.number_of_vehicles.shift(5)
        full_data["arac_sayi(t-6)"]=full_data.number_of_vehicles.shift(6)
        full_data["arac_sayi(t-12)"]=full_data.number_of_vehicles.shift(12)
        #full_data["arac_sayi(t-16)"]=full_data.number_of_vehicles.shift(16)
        full_data["arac_sayi(t-24)"]=full_data.number_of_vehicles.shift(24)
        #full_data["arac_sayi(t-36)"]=full_data.number_of_vehicles.shift(36)
        full_data["arac_sayi(t-48)"]=full_data.number_of_vehicles.shift(48)


        full_data["arac_sayi(t-1)"].fillna(method="bfill",inplace=True)
        full_data["arac_sayi(t-2)"].fillna(method="bfill",inplace=True)
        full_data["arac_sayi(t-3)"].fillna(method="bfill",inplace=True)
        full_data["arac_sayi(t-4)"].fillna(method="bfill",inplace=True)
        full_data["arac_sayi(t-5)"].fillna(method="bfill",inplace=True)
        full_data["arac_sayi(t-6)"].fillna(method="bfill",inplace=True)
        full_data["arac_sayi(t-12)"].fillna(method="bfill",inplace=True)
        #full_data["arac_sayi(t-16)"].fillna(method="bfill",inplace=True)
        full_data["arac_sayi(t-24)"].fillna(method="bfill",inplace=True)
        #full_data["arac_sayi(t-36)"].fillna(method="bfill",inplace=True)
        full_data["arac_sayi(t-48)"].fillna(method="bfill",inplace=True)

        full_data.loc[(full_data["date_time"]>="2020-04-24") & (full_data["date_time"]<="2020-05-26"),"ramazan"]=1
        full_data.ramazan=full_data.ramazan.fillna(0)

        full_data.loc[(full_data["date_time"]>="2020-07-31") & (full_data["date_time"]<="2020-08-03"),"kurban"]=1
        full_data.kurban=full_data.kurban.fillna(0)

        full_data=full_data.sort_values(by="date_time").reset_index(drop=True)

        return full_data

    @st.cache
    def split_data(full_data):

        test=full_data[full_data.month==12]
        train=full_data[(full_data.month<12)]

        X_train=train.drop(["date_time","number_of_vehicles"],axis=1)
        y_train=train["number_of_vehicles"]
        X_test=test.drop(["date_time","number_of_vehicles"],axis=1)
        y_test=test["number_of_vehicles"]

        return X_train,y_train,X_test,y_test

    @st.cache
    def scale_data(X_train,X_test):
        scaler=MinMaxScaler()
        X_train_scaled=scaler.fit_transform(X_train)
        X_test_scaled=scaler.transform(X_test)

        return X_train_scaled,X_test_scaled


    def get_model_summary(model):

        stream = io.StringIO()
        model.summary(print_fn=lambda x: stream.write(x + '\n'))
        summary_string = stream.getvalue()
        stream.close()
        return summary_string


    def model():

        
        number_of_hidden=st.number_input("Number of Hidden Layers",min_value=1,max_value=10)
        input_layer_neuron=st.slider("Number of Neurons in Input Layer",min_value=1,max_value=500)

        layer_neurons=[]

        for i in range(1,number_of_hidden+1):
            globals()["neuron_layer_" + str(i)]=st.slider(f"Number of Neurons in Layer {i}",min_value=1,max_value=500)
            layer_neurons.append( globals()["neuron_layer_" + str(i)])


        model=models.Sequential()

        input_shape= X_train_scaled.shape[1]

        model.add(layers.Dense(input_layer_neuron,input_shape=(input_shape,)))
        model.add(activation_func)

        for i in layer_neurons:

            model.add(layers.Dense(i))
            model.add(activation_func)

        model.add(layers.Dense(1))
        model.add(activation_func)

        model.compile(optimizer=opt, 
                        loss='mse',
                        metrics=["mean_squared_error"])
        return model


    def fit_model(model):

        
        start=time.time()
        placeholder=st.empty()
        placeholder.write("Training...")
    
        history=model.fit(X_train_scaled,y_train, 
                    epochs=epoch, 
                    verbose=1,
                    shuffle=False,
                    validation_data=(X_test_scaled,y_test),
                    batch_size=batch_size)
        
        end=time.time()

        model_time=end-start

        placeholder.success(f"Finished! Training Time : {model_time} ")


        y_pred = model.predict(X_test_scaled) 

        test_date=test.date_time
        
        
        result=pd.DataFrame({"date":test_date,"test":y_test,"pred":y_pred.ravel()})
        result=result.set_index(result.date).drop("date",axis=1)

        result.pred=result.pred.astype("int64")
        

        loss_df=pd.DataFrame({"train_loss":history.history["loss"],"val_loss":history.history["val_loss"]})

        return y_pred,result,loss_df


    def show_results():
        st.table(error_table(y_pred,y_test))
        fig = px.line(data_frame=result,
                            title="Prediction Plot",
                            labels={"value":"Number of Vehicles","date":"Date"})

        st.plotly_chart(fig)

        
        fig=px.line(loss_df,title="Loss Function",
                        labels={"index":"Epoch","value":"MSE"})

        st.plotly_chart(fig)
        st.markdown("Last 20 Prediction")
        st.write(result.tail(20))


    def create_feats_and_conv_date():
        frame_list=[]
        for i in range(1,17):
            globals()["region_" + str(i)]=read_files(f"region_{i}.csv")
            frame_list.append( globals()["region_" + str(i)])

        for i in range(0,16):
            convert_to_date(frame_list[i])
        return frame_list

    @st.cache(allow_output_mutation=True)
    def plot_vehicle_nos(frame,title):

        fig = px.line(data_frame=frame,
                        x="date_time",
                        y="number_of_vehicles",title=title)
        
        return fig

    
    if page_choice=="Plots":
        
        
        frame_list=create_feats_and_conv_date()
        loc_data,m=create_map()

        coord_list=[]
        for i in loc_data.values.tolist():
            i=str(i)
            coord_list.append(i)
    
        choice=st.selectbox(label="Coordinate",
                        options=coord_list)
        
        for i in range(0,16):

            if choice==coord_list[i]:

                frame=frame_list[i]
                min_value=frame["date_time"].min()
                max_value=frame["date_time"].max()
                st.write("Start Date: ",min_value,"End Date: ",max_value)

                date_time=st.date_input("Choose a date",min_value=min_value,
                                                max_value=max_value,value=min_value)
            
                tomorrow=date_time + datetime.timedelta(days=1)
                date_time=datetime.datetime.strftime(date_time,'%Y-%m-%d') #'%Y-%m-%d %H:%M:%S'  
                frame_to_plot=frame[(frame["date_time"]>=date_time) & (frame["date_time"].dt.date< tomorrow)]

                st.write(frame_to_plot)
                st.write(frame_to_plot[["minimum_speed","maximum_speed","average_speed","number_of_vehicles"]].describe().T)
                loc=json.loads(choice)

                m = folium.Map(loc, zoom_start=5,width="%100", height="%100")
                folium.Marker(loc, popup =f'{loc}' ).add_to(m)
                folium_static(m)
                fig=plot_vehicle_nos(frame_to_plot,f"Area {i+1} - Date: {date_time}")
                st.plotly_chart(fig)

                fig=plt.figure(figsize=(8,3))
                sns.distplot(frame_to_plot["number_of_vehicles"])
                st.pyplot(fig)

                fig=plt.figure(figsize=(8,3))
                sns.boxplot(frame_to_plot["number_of_vehicles"])
                st.pyplot(fig)
            

            
    if page_choice=="Codes":

        st.title("Codes")
        st.code(open("app.py").read())



    if page_choice=="Prediction":

        st.title("Coordinates")

        loc_data,m=create_map()
        folium_static(m)

        frame_list=create_feats_and_conv_date()

        weather_cond=read_files("Besiktas.csv").reset_index()
        weather_cond.date_time=pd.to_datetime(weather_cond.date_time)

        coord_list=[]
        for i in loc_data.values.tolist():
            i=str(i)
            coord_list.append(i)
    
        choice=st.selectbox(label="Coordinate",
                        options=coord_list)
                            

        for i in range(0,16):
            if choice==coord_list[i]:
                full_data=prepare_data(frame_list[i],weather_cond)

        st.markdown("#### Data (First 5)")
        st.dataframe(full_data.head())
        #st.markdown(f"_Shape_: _{full_data.shape}_")

        cols=["minimum_speed","maximum_speed","average_speed","WindGustKmph","cloudcover","humidity",
        "maxtempC","mintempC","precipMM","tempC","hour","dayofweek","dayofmonth","is_weekend","arac_sayi(t-1)",
        "arac_sayi(t-2)","arac_sayi(t-3)","arac_sayi(t-4)","arac_sayi(t-5)","arac_sayi(t-6)","arac_sayi(t-12)","arac_sayi(t-24)",
        "arac_sayi(t-48)","ramazan","kurban"]
        
        
        selected_cols=st.multiselect("Select Columns",cols,default=cols)

        selected_cols.append("month")
        selected_cols.append("date_time")
        selected_cols.append("number_of_vehicles")
    
        full_data=full_data[selected_cols]
        
        test=full_data[full_data.month==12]
        train=full_data[(full_data.month<12)]
    

        X_train,y_train,X_test,y_test = split_data(full_data)
        X_train_scaled,X_test_scaled=scale_data(X_train,X_test)

        st.title("Hyperparameters")

        optimizer=st.selectbox("Optimization Algorithm",["Adam","Adadelta","Adagrad","SGD","RMSProp"])

        if optimizer=="Adam":

            lr=st.number_input("Learning Rate",0.001,0.5,step=0.001,format="%f")
            beta_1=st.number_input("beta_1",0.1,1.0,step=0.01) #0.9
            beta_2=st.number_input("beta_2",0.1,1.0,step=0.01)  #0.999
            epsilon=1e-07

            st.write(f"lr : {lr} beta_1 : {beta_1} beta_2 : {beta_2} epsilon : {epsilon}")
            opt=Adam(learning_rate=lr,
                        beta_1=beta_1,
                        beta_2=beta_2,
                        epsilon=epsilon)
                        

        if optimizer =="Adadelta":
            lr=st.number_input("Learning Rate",0.001,0.5,step=0.001,format="%f")
            rho=st.number_input("rho",0.1,1.0)
            epsilon=1e-07
            st.write(f"lr : {lr} rho : {rho} epsilon : {epsilon} ")
            opt=Adadelta(learning_rate=lr,
                            rho=rho,
                            epsilon=epsilon)


        if optimizer=="Adagrad":

            lr=st.number_input("Learning Rate",0.001,0.5,step=0.001,format="%f")
            init_acc_value=st.number_input("Initial Accumulature Value",0.1,1.0,format="%f")
            epsilon=1e-07
            st.write(f"lr : {lr} initial accumulator value : {init_acc_value} epsilon : {epsilon} ")
            opt=Adagrad(learning_rate=lr,
                            initial_accumulator_value=init_acc_value,
                            epsilon=epsilon)            

    
            

        if optimizer=="SGD":

            lr=st.number_input("Learning Rate",0.001,0.1,step=0.001,format="%f") #0.01
            momentum=st.number_input("Momentum",0.0,1.0,step=0.1) #0.0
            st.write(f"lr : {lr} momentum : {momentum}")
            opt=SGD(learning_rate=lr,
                        momentum=momentum)
        

        if optimizer=="RMSProp":
            lr=st.number_input("Learning Rate",0.001,0.5,step=0.001,format="%f") #0.001
            rho=st.number_input("rho",0.1,1.0)
            momentum=st.number_input("momentum",0.0,1.0,step=0.1)
            epsilon=1e-07
            st.write(f"lr : {lr} rho : {rho} momentum : {momentum} epsilon : {epsilon}")
            opt=RMSprop(learning_rate=lr,
                            rho=rho,
                            momentum=momentum,
                            epsilon=epsilon)

        epoch=st.slider("Epoch",1,100)
        batch_size=st.slider("Batch Size",0,500,step=4)
        activation_func=st.selectbox("Activation Function",["ReLU","Leaky ReLU","ELU"])

        if activation_func=="ReLU":

            activation_func=ReLU()

        if activation_func=="Leaky ReLU":

            activation_func=LeakyReLU()

        if activation_func=="ELU":

            activation_func=ELU()

    

        model=model()

        st.subheader("Model")

        st.text(get_model_summary(model))
    

        st.info("Press the button again if you encounter an unexpected result!")
        
    
        
        if st.button("Predict"):

        
            st.title("Prediction Results")
            y_pred,result,loss_df=fit_model(model) 

            
            show_results()
            result["datetime"]=result.index
            
            csv = result.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()  

            linko= f'<a href="data:file/csv;base64,{b64}" download="result.csv">Download Result as CSV File</a>'

            st.markdown(linko, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
        
    
            
            
        
            
        
        

