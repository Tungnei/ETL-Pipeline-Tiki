import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import ollama
import google.generativeai as genai
from cachetools import cached, TTLCache

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
# path = 'D:\Source_project\ETL_Pipeline_Tiki\data_crawl'
# files = os.listdir(path)
# files_list = [os.path.join(path, file) for file in files if file.endswith('.csv')]
# all_dataframes = []
# for file in files_list:
#         df = pd.read_csv(file)
#         all_dataframes.append(df)
# data = pd.concat(all_dataframes, ignore_index=True)
# df = data[['id', 'product_name', 'brand_name', 'original_price', 'price_after_voucher', 'discount_rate', 'discount_price', 'quantity_sold', 'rating_average', 'review_count', 'warranty_info', 'return_policy', 'date']].copy()

# df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
# df = df.sort_values(by='date')
# df['quantity_sold_by_day'] = df.groupby('id')['quantity_sold'].diff().fillna(0).astype(int)
# df['quantity_sold_by_day'] = df['quantity_sold_by_day'].apply(lambda x: 0 if x < 0 else x)
# df['sale_by_day'] = df['quantity_sold_by_day'] * df['price_after_voucher']
# df = df[['id', 'product_name', 'brand_name', 'original_price', 'price_after_voucher', 'discount_rate', 'discount_price', 'rating_average', 'review_count', 'warranty_info', 'return_policy', 'date', 'quantity_sold', 'quantity_sold_by_day', 'sale_by_day']].copy()


# Mã nhúng Power BI (iframe)
powerbi_embed_code = '''
    <iframe 
        title="DashBoard-Tiki"
        center="true" 
        width="1140" 
        height="541.25" 
        src="https://app.powerbi.com/reportEmbed?reportId=8d3b50e1-ff3e-4449-b210-e688c20d0180&autoAuth=true&ctid=e7572e92-7aee-4713-a3c4-ba64888ad45f" 
        frameborder="0" 
        allowFullScreen="true">
    </iframe>
'''

# Thiết lập lựa chọn trang với radio button
page = st.sidebar.selectbox("Lựa chọn hiển thị", ["Dashboard Tiki", "Random Forest Model","Chatbot"])

if page == "Dashboard Tiki":
    # Trang 1: Hiển thị Dashboard Power BI
    st.markdown(powerbi_embed_code, unsafe_allow_html=True)
   
elif page == "Random Forest Model":
    # Trang 2: Hiển thị kết quả mô hình ML
    st.write("### Kết quả đánh giá mô hình Random Forest")

    # Đọc và xử lý dữ liệu
    path = 'D:\Source_project\ETL_Pipeline_Tiki\data_crawl'
    files = os.listdir(path)
    files_list = [os.path.join(path, file) for file in files if file.endswith('.csv')]
    all_dataframes = []
    for file in files_list:
            df = pd.read_csv(file)
            all_dataframes.append(df)
    data = pd.concat(all_dataframes, ignore_index=True)
    df = data[['id', 'product_name', 'brand_name', 'original_price', 'price_after_voucher', 'discount_rate', 'discount_price', 'quantity_sold', 'rating_average', 'review_count', 'warranty_info', 'return_policy', 'date']].copy()

    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df = df.sort_values(by='date')
    df['quantity_sold_by_day'] = df.groupby('id')['quantity_sold'].diff().fillna(0).astype(int)
    df['quantity_sold_by_day'] = df['quantity_sold_by_day'].apply(lambda x: 0 if x < 0 else x)
    df['sale_by_day'] = df['quantity_sold_by_day'] * df['price_after_voucher']
    df = df[['id', 'product_name', 'brand_name', 'original_price', 'price_after_voucher', 'discount_rate', 'discount_price', 'rating_average', 'review_count', 'warranty_info', 'return_policy', 'date', 'quantity_sold', 'quantity_sold_by_day', 'sale_by_day']].copy()


    original_data = df
    threshold = 1000  # Ngưỡng để xác định sản phẩm "hot"
    original_data['is_hot'] = original_data['quantity_sold'].apply(lambda x: 1 if x > threshold else 0)

    # Chọn các cột đặc trưng và nhãn
    X_class = original_data[['price_after_voucher', 'discount_rate', 'rating_average', 'review_count', 'sale_by_day']]
    y_class = original_data['is_hot']

    # Chia dữ liệu thành bộ huấn luyện và kiểm tra
    X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

    # Khởi tạo mô hình Random Forest
    rf_model = RandomForestClassifier(random_state=42, n_estimators=100)

    # Huấn luyện mô hình
    rf_model.fit(X_train_class, y_train_class)

    # Dự đoán trên bộ kiểm tra
    y_pred_rf = rf_model.predict(X_test_class)

    # Tính toán độ chính xác và AUC-ROC
    accuracy_rf = accuracy_score(y_test_class, y_pred_rf)
    roc_auc_rf = roc_auc_score(y_test_class, y_pred_rf)

    # Hiển thị kết quả
    st.write(f'Accuracy: {accuracy_rf: f}')
    st.write(f'AUC-ROC: {roc_auc_rf: f}')

    # Lấy độ quan trọng của các đặc trưng từ mô hình Random Forest
    feature_importance = rf_model.feature_importances_

    # Tạo dataframe để dễ dàng vẽ biểu đồ
    features = X_class.columns
    importance_df = pd.DataFrame({'feature': features, 'importance': feature_importance})

    # Sắp xếp theo độ quan trọng giảm dần
    importance_df = importance_df.sort_values(by='importance', ascending=False)

    # Tạo một biểu đồ với các cải tiến đẹp mắt
    plt.figure(figsize=(12, 8))
    bars = plt.barh(importance_df['feature'], importance_df['importance'], 
                    color=plt.cm.Blues(importance_df['importance'] / max(importance_df['importance'])),
                    edgecolor='grey')

    # Thêm nhãn vào mỗi thanh
    for bar in bars:
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                 f'{bar.get_width(): f}', va='center', ha='left', fontsize=12, color='black')

    # Tinh chỉnh nhãn trục x và trục y
    plt.xlabel('Importance', fontsize=14)
    plt.title('Feature Importance of Random Forest Model', fontsize=16)
    plt.gca().invert_yaxis()  # Đảo ngược trục y để đặc trưng quan trọng nhất lên đầu

    # Tinh chỉnh font chữ và không gian
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()

    # Hiển thị biểu đồ trong Streamlit
    st.pyplot(plt)

    # Nhóm dữ liệu theo ID sản phẩm và tính tổng số lượng bán cho mỗi sản phẩm
    top_hot_products_grouped = original_data.groupby(['id', 'product_name', 'brand_name', 'price_after_voucher']).agg({
        'quantity_sold': 'sum',
        'rating_average': 'mean',
        'review_count': 'sum'
    }).reset_index()

    # Sắp xếp theo số lượng bán (quantity_sold) để lấy 5 sản phẩm "hot" nhất
    top_hot_products_sorted_grouped = top_hot_products_grouped.sort_values(by='quantity_sold', ascending=False).head(5)

    # Hiển thị chi tiết 5 sản phẩm hot nhất
    st.write("### 5 Sản phẩm hot nhất:")
    st.dataframe(top_hot_products_sorted_grouped[['id', 'product_name', 'brand_name', 'price_after_voucher', 'quantity_sold', 'rating_average', 'review_count']])

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



