import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import pmdarima as pm  # Thư viện tự động chọn tham số ARIMA
import seaborn as sns
import requests
import google.generativeai as genai
from cachetools import cached, TTLCache
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from datetime import timedelta
from sklearn.linear_model import LinearRegression
import webbrowser

# Thiết lập lựa chọn trang với radio button
page = st.sidebar.selectbox("Lựa chọn hiển thị", ["Dashboard Tiki", "Random Forest Model","KNN","Linear  Regression","ARIMA","Chatbot"])

FASTAPI_URL = "http://127.0.0.1:8000/dashboard"
response = requests.get(FASTAPI_URL)
# Điều hướng tới trang Dashboard nếu chọn "Dashboard Tiki"
if page == "Dashboard Tiki":
    if response.status_code == 200:
    # Hiển thị nội dung HTML trong Streamlit
        webbrowser.open("http://127.0.0.1:8000/dashboard")
        st.write("Redirecting to the dashboard...")
elif page == "Random Forest Model":
    
    # Hàm tải và tiền xử lý dữ liệu
    @st.cache_data
    def load_and_process_data(path):
        files = os.listdir(path)
        files_list = [os.path.join(path, file) for file in files if file.endswith('.csv')]
        all_dataframes = []
        for file in files_list:
            df = pd.read_csv(file)
            all_dataframes.append(df)
        data = pd.concat(all_dataframes, ignore_index=True)

        df = data[['id', 'product_name', 'brand_name', 'original_price', 'price_after_voucher',
                'discount_rate', 'discount_price', 'quantity_sold', 'rating_average',
                'review_count', 'warranty_info', 'return_policy', 'date']].copy()

        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
        df = df.sort_values(by='date')
        df['quantity_sold_by_day'] = df.groupby('id')['quantity_sold'].diff().fillna(0).astype(int)
        df['quantity_sold_by_day'] = df['quantity_sold_by_day'].apply(lambda x: 0 if x < 0 else x)
        df['sale_by_day'] = df['quantity_sold_by_day'] * df['price_after_voucher']
        df['is_hot'] = df['quantity_sold'].apply(lambda x: 1 if x > 1000 else 0)  # Ngưỡng sản phẩm "hot"

        return df


    # Load dữ liệu
    data_path = 'D:/Source_project/ETL_Pipeline_Tiki/data_crawl'
    original_data = load_and_process_data(data_path)

    # Chọn các cột đặc trưng và nhãn
    X_class = original_data[['price_after_voucher', 'discount_rate', 'rating_average', 'review_count', 'sale_by_day']]
    y_class = original_data['is_hot']

    # Chia dữ liệu thành bộ huấn luyện và kiểm tra
    X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

    # Huấn luyện mô hình Random Forest
    @st.cache_data
    def train_random_forest():
        rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
        rf_model.fit(X_train_class, y_train_class)
        return rf_model

    rf_model = train_random_forest()

    # Tạo Tabs
    tabs = st.tabs(["Phân tích mô hình", "Dự đoán sản phẩm HOT", "Phân tích tương quan"])

    # Tab 1: Phân tích mô hình
    with tabs[0]:
        st.write("### Kết quả Đánh giá Mô hình Random Forest")
        y_pred_rf = rf_model.predict(X_test_class)
        accuracy_rf = accuracy_score(y_test_class, y_pred_rf)
        roc_auc_rf = roc_auc_score(y_test_class, y_pred_rf)
        st.write(f"Độ chính xác: {accuracy_rf:.2f}")
        st.write(f"AUC-ROC: {roc_auc_rf:.2f}")

        st.write("### Độ quan trọng của các đặc trưng")
        feature_importance = rf_model.feature_importances_
        importance_df = pd.DataFrame({'feature': X_class.columns, 'importance': feature_importance})
        importance_df = importance_df.sort_values(by='importance', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=importance_df, palette="Blues_d")
        plt.title('Feature Importance', fontsize=16)
        st.pyplot(plt)

        # Hiển thị top 5 sản phẩm hot
        st.write("### Top 5 Sản phẩm HOT nhất")
        top_hot_products = original_data[original_data['is_hot'] == 1]

        # Sắp xếp theo số lượng bán (quantity_sold) giảm dần và loại bỏ các tên sản phẩm trùng lặp
        top_hot_products_sorted = (
            top_hot_products
            .sort_values(by='quantity_sold', ascending=False)
            .drop_duplicates(subset='product_name')  # Loại bỏ các sản phẩm trùng tên
            .head(5)  # Lấy top 5 sản phẩm
        )

    # Hiển thị kết quả
        st.dataframe(top_hot_products_sorted[['product_name', 'brand_name', 'price_after_voucher', 'quantity_sold', 'rating_average']])
        # Tab 2: Dự đoán sản phẩm HOT
    with tabs[1]:
        st.write("### Dự đoán sản phẩm mới")
        input_data = {
            'price_after_voucher': st.number_input("Giá sau voucher:", min_value=0.0, value=1000000.0, step=100000.0),
            'discount_rate': st.slider("Tỷ lệ giảm giá:", min_value=0, max_value=100, value=10),
            'rating_average': st.slider("Đánh giá trung bình:", min_value=0.0, max_value=5.0, value=4.5, step=0.1),
            'review_count': st.number_input("Số lượt đánh giá:", min_value=0, value=100),
            'sale_by_day': st.number_input("Doanh số bán theo ngày (VNĐ):", min_value=0.0, value=50000000.0)
        }
        input_df = pd.DataFrame([input_data])
        prediction = rf_model.predict(input_df)[0]
        st.write(f"Kết quả dự đoán: {'Sản phẩm HOT' if prediction == 1 else 'Sản phẩm không HOT'}")

    # Tab 3: Phân tích tương quan
    with tabs[2]:
        st.write("### Phân tích tương quan")
        corr_matrix = original_data[['price_after_voucher', 'discount_rate', 'rating_average', 'review_count', 'sale_by_day']].corr()
        st.write("Ma trận tương quan:")
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        st.pyplot(plt)



elif page == "Chatbot":
 
    class TikiGeminiChatbot:
        def __init__(self, data_path):
            # Configure Gemini API
            try:
                # Replace with your actual API key or use st.secrets for better security
                genai.configure(api_key='AIzaSyC3fgxOlIeU1DAWDWzTvpEHlR33l5MZpu4')
                self.model = genai.GenerativeModel('gemini-pro')
            except Exception as e:
                st.error(f"Lỗi kết nối Gemini: {e}")
                self.model = None

            # Load and preprocess data
            self.df = self.load_and_preprocess_data(data_path)
            self.df_summary = self.df.head(50) # Limiting the preview to top 5 rows for efficiency

        def load_and_preprocess_data(self, path):
            try:
                df = pd.read_csv(path)
                # Optional: You can perform additional preprocessing steps here
                # df['date'] = pd.to_datetime(df['date'], errors='coerce')
                # df = df.dropna(subset=['date'])  # Remove rows with invalid dates if needed
                self.df = df
                return df
            except Exception as e:
                st.error(f"Lỗi khi tải và xử lý dữ liệu: {e}")
                return pd.DataFrame()  # Return an empty DataFrame in case of error

        def get_gemini_answer(self, user_input):
            try:
                # Prepare context with sample data
                context = f"""
                Bạn là trợ lý AI chuyên phân tích dữ liệu sản phẩm Tiki. 
                Dưới đây là một số sản phẩm trong tập dữ liệu:
                {self.df_summary.to_string(index=False)}
                
                Hãy trả lời câu hỏi sau dựa trên bối cảnh và dữ liệu trên:
                Câu hỏi: {user_input}
                """

                # Generate response from Gemini API
                if self.model:
                    response = self.model.generate_content(context)
                    return response.text

                return "Gemini AI không khả dụng."
            except Exception as e:
                return f"Đã có lỗi xảy ra khi truy vấn Gemini: {e}"

        def local_dataframe_search(self, user_input):
            try:
                keywords = user_input.lower().split()
                relevant_rows = self.df_summary[
                    self.df_summary.apply(lambda row: any(keyword in str(row).lower() for keyword in keywords), axis=1)
                ]
                
                if not relevant_rows.empty:
                    return f"Dựa trên dữ liệu, tôi tìm thấy các sản phẩm liên quan:\n{relevant_rows.to_string(index=False)}"
                else:
                    return "Xin lỗi, tôi không tìm thấy thông tin phù hợp với câu hỏi của bạn."
            except Exception as e:
                return f"Đã có lỗi khi tìm kiếm dữ liệu: {e}"

        def get_answer(self, user_input):
            # First try Gemini AI
            if self.model:
                answer = self.get_gemini_answer(user_input)
                if answer:
                    return answer

            # Fallback to local search
            return self.local_dataframe_search(user_input)

    def chatbot_page():
        # Đường dẫn thư mục chứa dữ liệu
        path = 'D:/Source_project/ETL_Pipeline_Tiki/data_gemini.csv'
        
        # Initialize Chatbot
        chatbot = TikiGeminiChatbot(path)

        # Giao diện Streamlit
        st.title("Chatbot Hỏi Đáp về Dữ Liệu Tiki")

        # Hộp văn bản cho người dùng nhập câu hỏi
        user_input = st.text_area("Nhập câu hỏi của bạn:")

        if st.button("Run query"):
                with st.spinner("Processing..."):
        # Xử lý câu hỏi và hiển thị câu trả lời
                    if user_input:
            
            # Lấy câu trả lời từ chatbot
                        response = chatbot.get_answer(user_input)
            
            # Hiển thị câu trả lời
                        st.write(f"Chatbot: {response}")

    def main():
        # Sidebar để chọn trang
        # page = st.sidebar.selectbox("Choose a page", ["Home", "Chatbot"])
        
        # Hiển thị trang Chatbot nếu người dùng chọn
        if page == "Chatbot":
            chatbot_page()
        else:
            st.write("Welcome to the Tiki Data Analysis App")

    # Call the main function to run the app
    if __name__ == "__main__":
        main()


elif page == "KNN":
  
    model_knn = 'D:\Source_project\ETL_Pipeline_Tiki\model\knn_final_model2.pkl'
    model = joblib.load(model_knn)

    #  sacler.pkl
    st.title('Dự đoán phân khúc sản phẩm')

    rating_average = st.number_input('Đánh giá trung bình', min_value=0.0, max_value=5.0, value=4.0, step=0.1)
    price_after_voucher = st.number_input('Giá sau voucher (VND)', min_value=0, value=3000000)
    review_count = st.number_input('Số lượng đánh giá', min_value=0, value=100)
    quantity_sold = st.number_input('Số lượng bán', min_value=0, value=500)
    discount_rate = st.number_input('Tỷ lệ giảm giá(%)', min_value=0, max_value=80, value=1, step=1)

    if st.button('Dự đoán'):
        new_data = pd.DataFrame({
        'rating_average': [rating_average],
        'price_after_voucher': [price_after_voucher],
        'review_count': [review_count],
        'quantity_sold': [quantity_sold],
        'discount_rate': [discount_rate]
         })

    #  chuẩn hóa dữ liệu
        scaler = StandardScaler()
        new_data_scaled = scaler.fit_transform(new_data)

        prediction = model.predict(new_data_scaled)
        st.subheader(f'Kết quả Phân loại sản phẩm: {prediction[0]}')

elif page == "ARIMA":
   

    # Bước 1: Tải dữ liệu sản phẩm
    df_clean = pd.read_csv('D:/Source_project/ETL_Pipeline_Tiki/model.csv')  # Đường dẫn tới file CSV
    df_clean['date'] = pd.to_datetime(df_clean['date'])
    df_clean.sort_values('date', inplace=True)

    # Bước 2: Giao diện người dùng
    st.title('Dự báo giá, Doanh thu sản phẩm theo thời gian (đơn vị: triệu đồng)')

    # Lựa chọn sản phẩm từ dữ liệu
    product_choices = df_clean['product_name'].unique()  # Giả sử cột 'product_name' chứa tên sản phẩm
    selected_product = st.selectbox('Chọn sản phẩm', product_choices)

    # Bước 3: Lọc dữ liệu của sản phẩm đã chọn
    product_data = df_clean[df_clean['product_name'] == selected_product]

    # Hiển thị thông tin sản phẩm
    st.write(f"Thông tin sản phẩm: {selected_product}")
    st.write(product_data.tail(1))

    # Bước 4: Tạo mô hình ARIMA cho giá sản phẩm
    price_data = product_data[['date', 'price_after_voucher']]
    price_data.set_index('date', inplace=True)

    # Chuyển đổi giá sang đơn vị "triệu đồng"
    price_data['price_after_voucher'] = price_data['price_after_voucher'] / 1_000_000

    # Xây dựng mô hình ARIMA (p=1, d=1, q=1 là tham số mẫu, bạn có thể tinh chỉnh)
    model = ARIMA(price_data['price_after_voucher'], order=(1, 1, 1))
    model_fit = model.fit()

    # Bước 5: Nhập số ngày dự báo
    forecast_days = st.number_input('Nhập số ngày bạn muốn dự báo', min_value=1, max_value=30, value=7)
    tab = st.tabs(["Dự báo giá sản phẩm", "Dự báo Doanh thu"])
    # Bước 6: Dự báo giá sản phẩm
    with tab[0]:
        forecast = model_fit.forecast(steps=forecast_days)
        forecast_dates = pd.date_range(start=price_data.index[-1] + timedelta(days=1), periods=forecast_days, freq='D')

        # Hiển thị kết quả dự báo
        forecast_df = pd.DataFrame({
            'Ngày dự báo': forecast_dates,
            'Giá dự báo (triệu đồng)': forecast
        })
        st.write(forecast_df)

        # Bước 7: Vẽ đồ thị kết quả
        plt.figure(figsize=(10, 6))

        # Vẽ giá thực tế
        plt.plot(price_data.index, price_data['price_after_voucher'], label='Giá thực tế (triệu đồng)', color='blue')

        # Vẽ giá dự báo
        plt.plot(forecast_dates, forecast, label='Giá dự báo (triệu đồng)', color='red', linestyle='--')

        # Thêm nhãn và tiêu đề
        plt.title(f'Dự báo giá sản phẩm: {selected_product}', fontsize=16)
        plt.xlabel('Ngày', fontsize=14)
        plt.ylabel('Giá (triệu đồng)', fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Hiển thị đồ thị trên Streamlit
        st.pyplot(plt)
    with tab[1]:
        # Tính toán doanh thu từ giá và số lượng bán
        product_data['revenue'] = product_data['price_after_voucher'] * product_data['quantity_sold_by_day']

        # Chuyển đổi doanh thu sang triệu đồng
        product_data['revenue_million'] = product_data['revenue'] / 1_000_000

        # Tạo series doanh thu
        revenue_data = product_data[['date', 'revenue_million']]
        revenue_data.set_index('date', inplace=True)

        # Xây dựng mô hình ARIMA cho doanh thu
        revenue_model = ARIMA(revenue_data['revenue_million'], order=(1, 1, 1))
        revenue_model_fit = revenue_model.fit()

        # Dự báo doanh thu
        revenue_forecast = revenue_model_fit.forecast(steps=forecast_days)
        revenue_forecast_dates = pd.date_range(start=revenue_data.index[-1] + timedelta(days=1), periods=forecast_days, freq='D')

        # Hiển thị kết quả dự báo doanh thu
        revenue_forecast_df = pd.DataFrame({
            'Ngày dự báo': revenue_forecast_dates,
            'Doanh thu dự báo (triệu đồng)': revenue_forecast
        })
        st.write("Dự báo Doanh thu:", revenue_forecast_df)

        # Vẽ đồ thị doanh thu
        plt.figure(figsize=(10, 6))

        # Vẽ doanh thu thực tế
        plt.plot(revenue_data.index, revenue_data['revenue_million'], 
                label='Doanh thu thực tế (triệu đồng)', color='green')

        # Vẽ doanh thu dự báo
        plt.plot(revenue_forecast_dates, revenue_forecast, 
                label='Doanh thu dự báo (triệu đồng)', color='red', linestyle='--')

        # Thêm nhãn và tiêu đề
        plt.title(f'Dự báo Doanh thu sản phẩm: {selected_product}', fontsize=16)
        plt.xlabel('Ngày', fontsize=14)
        plt.ylabel('Doanh thu (triệu đồng)', fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Hiển thị đồ thị doanh thu trên Streamlit
        st.pyplot(plt)

        # Thêm thống kê so sánh
        st.subheader("Phân tích so sánh")

        # Tính toán một số chỉ số thống kê
        avg_price = product_data['price_after_voucher'].mean()
        avg_revenue = product_data['revenue_million'].mean()
        max_revenue = product_data['revenue_million'].max()
        min_revenue = product_data['revenue_million'].min()

        # Hiển thị các chỉ số
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Giá trung bình", f"{avg_price:.2f} tr", "Giá")
        col2.metric("Doanh thu trung bình", f"{avg_revenue:.2f} tr", "Doanh thu")
        col3.metric("Doanh thu cao nhất", f"{max_revenue:.2f} tr", "Cao nhất")
        col4.metric("Doanh thu thấp nhất", f"{min_revenue:.2f} tr", "Thấp nhất")


elif page == "Linear  Regression":
        # Hàm dự báo giá
    def forecast_price(df, forecast_days=7):
        price_data = df[['date', 'price_after_voucher']]
        price_data.set_index('date', inplace=True)

        if len(price_data) < 10:
            st.warning("Không đủ dữ liệu để dự báo giá")
            return None, None, None

        model = pm.auto_arima(
            price_data['price_after_voucher'],
            seasonal=True,
            m=7,
            suppress_warnings=True,
            stepwise=True
        )

        forecast, conf_int = model.predict(n_periods=forecast_days, return_conf_int=True)

        forecast_dates = pd.date_range(
            start=price_data.index[-1] + timedelta(days=1),
            periods=forecast_days,
            freq='D'
        )

        forecast_df = pd.DataFrame({
            'Ngày dự báo': forecast_dates,
            'Giá dự báo (triệu đồng)': forecast / 1_000_000,
            'Khoảng tin cậy dưới (triệu đồng)': conf_int[:, 0] / 1_000_000,
            'Khoảng tin cậy trên (triệu đồng)': conf_int[:, 1] / 1_000_000
        })

        return model, forecast_df, price_data


    # Hàm dự báo doanh thu
    def forecast_revenue(df, forecast_days=7):
        revenue_data = df[['date', 'quantity_sold_by_day']]
        revenue_data.set_index('date', inplace=True)

        if len(revenue_data) < 10:
            st.warning("Không đủ dữ liệu để dự báo doanh thu")
            return None, None, None

        model = pm.auto_arima(
            revenue_data['quantity_sold_by_day'],
            seasonal=True,
            m=7,
            suppress_warnings=True,
            stepwise=True
        )

        forecast, conf_int = model.predict(n_periods=forecast_days, return_conf_int=True)

        forecast_dates = pd.date_range(
            start=revenue_data.index[-1] + timedelta(days=1),
            periods=forecast_days,
            freq='D'
        )

        forecast_df = pd.DataFrame({
            'Ngày dự báo': forecast_dates,
            'Doanh thu dự báo (triệu đồng)': forecast / 1_000_000,
            'Khoảng tin cậy dưới (triệu đồng)': conf_int[:, 0] / 1_000_000,
            'Khoảng tin cậy trên (triệu đồng)': conf_int[:, 1] / 1_000_000
        })

        return model, forecast_df, revenue_data


    # Hàm vẽ biểu đồ
    def visualize_forecast(data, forecast_df, label, y_label):
        plt.figure(figsize=(12, 6))

        # Vẽ dữ liệu thực tế
        plt.plot(data.index, data.iloc[:, 0] / 1_000_000, label=f'{label} thực tế (triệu đồng)', color='blue')

        # Vẽ dữ liệu dự báo
        plt.plot(forecast_df['Ngày dự báo'], forecast_df.iloc[:, 1], label=f'{label} dự báo (triệu đồng)', color='red', linestyle='--')

        # Vẽ khoảng tin cậy
        plt.fill_between(forecast_df['Ngày dự báo'], forecast_df.iloc[:, 2], forecast_df.iloc[:, 3], color='pink', alpha=0.3, label='Khoảng tin cậy 95%')

        plt.title(f'Dự báo {label}', fontsize=16)
        plt.xlabel('Ngày', fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        return plt


    # Hàm chính
    def main():
        # Tải dữ liệu
        df_clean = pd.read_csv('D:/Source_project/ETL_Pipeline_Tiki/model.csv')
        df_clean['date'] = pd.to_datetime(df_clean['date'])
        df_clean.sort_values('date', inplace=True)

        # Tiêu đề
        st.title('Dự báo Doanh thu và Giá bán sản phẩm (triệu đồng)')

        # Chọn sản phẩm
        product_choices = df_clean['product_name'].unique()
        selected_product = st.selectbox('Chọn sản phẩm', product_choices)

        # Lọc dữ liệu cho sản phẩm được chọn
        product_data = df_clean[df_clean['product_name'] == selected_product]

        # Số ngày dự báo
        forecast_days = st.number_input('Nhập số ngày dự báo', min_value=1, max_value=30, value=7)

        # Tạo tabs cho dự báo
        tab1, tab2 = st.tabs(["Dự báo Doanh thu", "Dự báo Giá bán"])

        with tab1:
            # Dự báo doanh thu
            revenue_model, revenue_forecast_df, revenue_data = forecast_revenue(product_data, forecast_days)
            if revenue_forecast_df is not None:
                st.write("Dự báo Doanh thu:", revenue_forecast_df)

                plt_revenue = visualize_forecast(revenue_data, revenue_forecast_df, 'Doanh thu', 'Doanh thu (triệu đồng)')
                st.pyplot(plt_revenue)

        with tab2:
            # Dự báo giá bán
            price_model, price_forecast_df, price_data = forecast_price(product_data, forecast_days)
            if price_forecast_df is not None:
                st.write("Dự báo Giá bán:", price_forecast_df)

                plt_price = visualize_forecast(price_data, price_forecast_df, 'Giá bán', 'Giá bán (triệu đồng)')
                st.pyplot(plt_price)


    # Chạy ứng dụng
    if __name__ == "__main__":
        main()