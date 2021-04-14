import streamlit as st

def description_page():      

   st.markdown("""


              **Data**: https://data.ibb.gov.tr/dataset/saatlik-trafik-yogunluk-veri-seti

            

              ## **Coordinates**

              **1**- [41.0421752929688, 29.0093994140625]

              **2**- [41.0476684570312, 29.0093994140625]

              **3**- [41.0366821289062, 28.9984130859375]

              **4**- [41.0531616210938, 29.0093994140625]
              
              **5**- [41.0586547851562, 29.0093994140625]

              **6**- [41.0476684570312, 28.9874267578125]

              **7**- [41.0421752929688, 28.9874267578125]

              **8**- [41.0366821289062, 28.9874267578125]

              **9**- [41.0311889648438, 28.9874267578125]

              **10**- [41.0586547851562, 28.9984130859375]

              **11**- [41.0641479492188, 29.0093994140625]

              **12**- [41.0476684570312, 29.0203857421875]

              **13**- [41.0421752929688, 28.9984130859375]
              
             **14**- [41.0476684570312, 28.9984130859375]

              **15**- [41.0531616210938, 28.9984130859375]

              **16**- [41.0641479492188, 28.9984130859375]

        
    """)

   st.image("images/besiktas.jpg",output_format="PNG",width=400)

   st.markdown("""
               ## **Variables**

               * date_time
               * minimum_speed
               * maximum_speed
               * average_speed
               * number_of_vehicles
               * WindGustKmph
               * cloudcover
               * humidity
               * maxtempC
               * mintempC
               * precipMM
               * tempC
               * hour
               * month
               * dayofweek
               * dayofmonth
               * is_weekend
               * arac_sayi(t-1)
               * arac_sayi(t-2)
               * arac_sayi(t-3)
               * arac_sayi(t-4)
               * arac_sayi(t-5)
               *arac_sayi(t-6)
               * arac_sayi(t-12)
               * arac_sayi(t-24)
               * arac_sayi(t-48)
               * ramazan
               * kurban


    """)