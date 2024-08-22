import streamlit as st
import pandas as pd
import pickle
from langdetect import detect
import re
from rapidfuzz import process, fuzz

# Hàm đề xuất 
def get_recommendations(df, hotel_id, cosine_sim, nums=5):
    # Get the index of the hotel that matches the hotel_id
    matching_indices = df.index[df['Hotel_ID'] == hotel_id].tolist()
    if not matching_indices:
        print(f"No hotel found with ID: {hotel_id}")
        return pd.DataFrame()  # Return an empty DataFrame if no match
    idx = matching_indices[0]

    # Get the pairwise similarity scores of all hotels with that hotel
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the hotels based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the nums most similar hotels (Ignoring the hotel itself)
    sim_scores = sim_scores[1:nums+1]

    # Get the hotel indices
    hotel_indices = [i[0] for i in sim_scores]

    # Return the top n most similar hotels as a DataFrame
    return df.iloc[hotel_indices]

# Hiển thị đề xuất ra bảng
def display_recommended_hotels(recommended_hotels, cols=5):
    for i in range(0, len(recommended_hotels), cols):
        cols = st.columns(cols)
        for j, col in enumerate(cols):
            if i + j < len(recommended_hotels):
                hotel = recommended_hotels.iloc[i + j]
                with col:   
                    st.write(hotel['Hotel_Name'])                    
                    expander_description = st.expander(f"Thông tin chi tiết")
                    hotel_description = hotel['Hotel_Description']
                    truncated_description = ' '.join(hotel_description.split()[:300]) + '...'
                    expander_description.write(truncated_description)
                    expander_description.markdown("Nhấn vào mũi tên để đóng hộp text này.")
                    
                    expander_Address = st.expander(f"Địa chỉ khách sạn")
                    hotel_description = hotel['Hotel_Address']
                    truncated_description = ' '.join(hotel_description.split())
                    expander_Address.write(truncated_description)
                    expander_Address.markdown("Nhấn vào mũi tên để đóng hộp text này.")           

#Lấy ra các khách sạn có mô tả tiếng việt
def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"


#Hàm tìm kiếm tên khách sạn
def search_hotel_name(df, search_string):
    if not search_string.strip():
        return pd.DataFrame()  # Trả về DataFrame trống nếu search_string rỗng

    # Tìm kiếm không phân biệt chữ hoa/chữ thường
    result_df = df[df['Hotel_Name'].str.contains(search_string, case=False, na=False)]
    return result_df

#Hàm tìm kiếm địa chỉ khách sạn
def search_hotel_address(df, search_string):
    if not search_string.strip():
        return pd.DataFrame()  # Trả về DataFrame trống nếu search_string rỗng

    # Tìm kiếm không phân biệt chữ hoa/chữ thường
    result_df = df[df['Hotel_Address'].str.contains(search_string, case=False, na=False)]
    return result_df

#Hàm tìm kiếm người dùng khách sạn
def search_Reviewer_name_idx(df, search_string):
    if not search_string.strip():
        return pd.DataFrame()  # Trả về DataFrame trống nếu search_string rỗng

    # Tìm kiếm không phân biệt chữ hoa/chữ thường
    result_df = df[df['Reviewer_name_idx'].str.contains(search_string, case=True, na=False)]
    return result_df

# Sử dụng cache để lưu dữ liệu dùng cho Content-based
@st.cache_data
def load_data():
    df = pd.read_csv('Data/hotel_info.csv', encoding="utf-8", delimiter=',')
    return df

# Sử dụng cache để lưu dữ liệu dùng cho Collaborative Filtering
@st.cache_data
def load_info():
    df_info = pd.read_csv('Data/hotel_info.csv')
    df_info = df_info[['Hotel_ID', 'Hotel_Name', 'Hotel_Address']]
    df_info.rename(columns={'Hotel_ID': 'Hotel ID'}, inplace=True)
    df_info.dropna()
    return df_info

# Sử dụng cache để lưu dữ liệu Collaborative Filtering
@st.cache_data
def load_data_final():
    df_hotels = pd.read_csv('Data/final_hotel.csv', encoding="utf-8", delimiter=',')
    return df_hotels

# Sử dụng cache để lưu cosine similarity
@st.cache_data
def load_cosine_similarity():
    with open('model/cosine.pkl', 'rb') as f:
        cosine_sim_new = pickle.load(f)
    return cosine_sim_new

# Sử dụng cache để lưu surprise_svd
@st.cache_data
def load_surprise_svd():
    with open('model/cosine.pkl', 'rb') as f:
        surprise_svd = pickle.load(f)
    return surprise_svd

#Tiền xử lý dữ liệu khách sạn
@st.cache_data
def data_pre(df):
    data_recommend = df[['Hotel_ID', 'Hotel_Name' ,'Hotel_Description', 'Hotel_Address']]
    data_recommend.drop_duplicates(inplace=True)
    data_recommend.dropna(inplace=True)
    data_recommend['Language'] = data_recommend['Hotel_Description'].apply(detect_language)
    data_recommend = data_recommend[data_recommend['Language'] == 'vi'].drop('Language', axis=1)
    return data_recommend

df_hotels = load_data_final()
df = load_data()
df_info = load_info()
data_recommend = data_pre(df)
cosine_sim_new = load_cosine_similarity()
surprise_svd = load_surprise_svd()

# Using menu
menu = ["Giới thiệu", "Xây dựng Collaborative Filtering", "Xây dựng Content-Based", "Đề xuất dựa trên người dùng", "Đề xuất dựa trên nội dung"]
choice = st.sidebar.selectbox('Danh mục', menu)
if choice == 'Giới thiệu':    
    st.image('Image/hotel.jpg', use_column_width=True)
    st.subheader("Tổng quan bài toán")
    st.subheader("1. Recommender system là gì?")
    st.write('Hệ thống gợi ý (Recommender System) là một công nghệ hoặc phương pháp được sử dụng để gợi ý sản phẩm, dịch vụ, hoặc nội dung cho người dùng dựa trên dữ liệu và hành vi của họ. Những hệ thống này thường được sử dụng trong các nền tảng như trang web thương mại điện tử, dịch vụ phát video, mạng xã hội, và nhiều ứng dụng khác để cải thiện trải nghiệm người dùng và tăng cường mức độ tương tác.')
    st.image('Image/rs.png')

    st.subheader("2. Collaborative Filtering (CF) và Content-Based")
    st.subheader("Collaborative Filtering")
    st.write('Gợi ý các sản phẩm, dịch vụ, nội dung cho một người dùng nào đó dựa trên mối quan tâm, sở thích (preferences) của những người dùng tương tự đối với các sản phẩm, dịch vụ, nội dung đó.')
    st.subheader("Có hai dạng chính của CF:")
    st.write('- User-based CF: Dự đoán sở thích của một người dùng dựa trên những người dùng khác có sở thích tương tự. Ví dụ, nếu người dùng A và B đều thích một nhóm sản phẩm giống nhau, và A thích một sản phẩm mới, thì sản phẩm đó có thể được đề xuất cho B.')
    st.write('- Item-based CF: Dự đoán sở thích của người dùng dựa trên sự tương đồng giữa các sản phẩm. Nếu người dùng đã thích sản phẩm A và B, và A và B tương tự nhau, thì sản phẩm C (tương tự A và B) sẽ được đề xuất cho người dùng.')
    st.subheader("Content-Based")
    st.write('Gợi ý các mục dựa trên đặc điểm của nội dung mà người dùng đã tương tác hoặc thể hiện sự quan tâm. Ví dụ, nếu bạn thích các bộ phim hành động, hệ thống sẽ gợi ý các bộ phim hành động khác.')

    st.subheader("3. Yêu cầu bài toán")
    st.write('Triển khai hệ thống Recommender System trên Agoda giúp đề xuất khách sạn/ resort phù hợp tới người dùng thông qua Content-Based và Collaborative Filtering')
    st.image('Image/agoda.png')
elif choice == 'Xây dựng Collaborative Filtering':
    st.subheader("Collaborative Filtering")
    hotel_comments = pd.read_csv('Data/hotel_comments.csv')
    st.write('Đọc dữ liệu hotel_comments')
    st.write('Tổng số dòng dữ liệu: ', len(hotel_comments))
    st.image('Image/CF/printSchema_comments.png')

    st.subheader("Tiền xử lý dữ liệu, lấy các cột cần thiết")
    st.write('Chuyển đổi cột Score thành kiểu Float')
    st.write('Drop_duplicate, dropna: ', len(df_hotels))
    st.write("Sử dụng Window trong pyspark.sql.window để tạo cột Reviewer_name_idx bằng cách row_number() với Reviewer Name ")
    st.image('Image/CF/windowspec.png')
    st.write('Lấy 3 cột Hotel_id, Reviewer_name_idx, Score')
    st.image('Image/CF/data_pre.png')

    st.subheader('Trực quan hóa dữ liệu')
    st.write('Top 10 khách sạn được đánh giá nhiều nhất')
    st.image('Image/CF/plt_hotel_id.png')
    st.write('Top 10 khách hàng đánh giá nhiều nhất')
    st.image('Image/CF/plt_reviewer_name_id.png')
    st.write('Biểu đồ phân phối số điểm đánh giá')
    st.image('Image/CF/plt_score.png')

    st.subheader('Xem các thông số và tính ma trận')
    st.write('Tổng số đánh giá: ', len(df_hotels))
    st.write('Số lượng khách hàng: ', len(df_hotels["Reviewer_name_idx"].drop_duplicates()))
    st.write('Số lượng khách sạn: ', len(df_hotels["Hotel ID"].drop_duplicates()))
    total_cells = len(df_hotels["Reviewer_name_idx"].drop_duplicates()) * len(df_hotels["Hotel ID"].drop_duplicates())
    st.write('Tổng số ô trong ma trận: ', total_cells)
    sparsity = 1 - (len(df_hotels)*1.0 / total_cells)
    st.write('Tỉ lệ còn trống khi đã được fill ban đầu: ', sparsity)

    st.subheader('Chuẩn hóa dữ liệu')
    st.write('Sử dụng StringIndexer để chuẩn hóa cho 2 cột Hotel ID và Reviewer_name_idx')
    st.image('Image/CF/final_data.png')

    st.subheader('===> Tiến hành xây dựng mô hình ALS và Surprise')

    st.subheader('Đánh giá mô hình')
    st.subheader('ALS')
    st.image('Image/CF/rmse_als.png')
    st.subheader('Surprise')
    st.image('Image/CF/rmse_surprise.png')

    st.subheader("Chọn SVD vì RMSE khá tốt kèm với tốc độ ổn định khi đưa lên GUI")
elif choice == 'Xây dựng Content-Based':
    st.subheader("Content-Based")
    st.write('Đọc dữ liệu hotel_info')
    st.write('Tổng số dòng dữ liệu: 740')
    st.image('Image/CB/info_printSchema.png')

    st.subheader("Tiền xử lý dữ liệu, lấy các cột cần thiết")
    st.write('Lấy các cột Hotel_ID, Hotel_Name, Hotel_Description, Hotel_Address')
    st.write('Drop_duplicate, dropna: 739')
    st.write('Sử dụng detect trong thư viện langdetect để lọc lấy các dòng tiếng việt trong Hotel_Description: 586')
    st.write("Loại bỏ cụm 'Nha Trang, Việt Nam...' trong Hotel_Address")
    st.image('Image/CB/description.png')
    st.write('Tạo cột Content bằng cách nối Hotel_Description và Hotel_Address')
    st.write('Sử dụng word_tokenize trong underthsea để tạo cột Content_wt')
    st.image('Image/CB/Content_wt.png')
    st.write('Đọc file vietnamese_stopwords để loại bỏ các stopword, ký tự đặc biệt, số trong Content_wt')
    st.image('Image/CB/Content_wt_stop.png')
    st.write('Tạo cột Content_pre bằng cách .split() Content_wt')
    st.write('Tạo dictionary bằng corpora.Dictionary với cột Content_pre: 5515')
    st.write('Tạo corpus bằng dictionary.doc2bow với cột Content_pre')
    st.image('Image/CB/corpus.png')

    st.subheader('Trực quan hóa dữ liệu')
    st.write('Tạo wordcloud cho Content_pre')
    st.image('Image/CB/wordcloud.png')

    st.subheader('===> Tiến hành xây dựng mô hình gensim và cosine')
    
    st.subheader('Đề xuất khách sạn')
    st.write('Đề xuất với gensim')
    st.image('Image/CB/gensim.png')
    st.write('Đề xuất với cosine')
    st.image('Image/CB/cosine.png')

    st.subheader('Chọn cosine vì các đề xuất tương đối khớp về địa chỉ với khách sạn đã chọn')

elif choice == 'Đề xuất dựa trên người dùng':
    st.subheader("Collaborative Filtering")
    st.subheader("Tìm tên người dùng")
    reviewer_name_idx = ''
    reviewer_name_idx = st.text_input("Nhập tên người dùng cần tìm:")
    st.subheader("Tên người dùng và các đánh giá của người dùng cho khách sạn")
    data = search_Reviewer_name_idx(df_hotels, reviewer_name_idx)
    if reviewer_name_idx == '':
        st.session_state.df_hotels = df_hotels
    elif data.empty:
        st.write('Không có kết quả với người dùng: ', '<' , reviewer_name_idx , '>')
        st.session_state.df_hotels = data
    else:
        st.write('Kết quả cho người dùng  ' , '<' , reviewer_name_idx , '>')
        st.session_state.df_hotels = data
        st.write('Người dùng ', '<', reviewer_name_idx ,'>', ' đã đánh giá cho',len(data), ' khách sạn')
    # Tạo một tuple cho mỗi khách sạn, trong đó phần tử đầu là tên và phần tử thứ hai là ID
    df_distinct = st.session_state.df_hotels.drop_duplicates(subset=['Reviewer_name_idx'])
    hotel_options = [(row['Reviewer_name_idx'], row['Reviewer_idx']) for index, row in st.session_state.df_hotels.drop_duplicates(subset=['Reviewer_name_idx']).iterrows()]
    st.session_state.df_hotels
    st.subheader("Hãy chọn 1 người dùng")
    # Tạo một dropdown với options là các tuple này
    selected_hotel = st.selectbox(
        "Chọn người dùng",
        options=hotel_options,
        format_func=lambda x: x[0]  # Hiển thị tên khách sạn
    )
    # Display the selected hotel
    st.write("Bạn đã chọn:", selected_hotel)
    st.subheader("Đề xuất cho bạn ")

    df_score = df_hotels[['Hotel_idx', 'Hotel ID']]
    df_score['EstimateScore'] = df_score['Hotel_idx'].apply(lambda x: surprise_svd.predict(selected_hotel[1], x).est) # est: get EstimateScore
    df_score = df_score.sort_values(by=['EstimateScore'], ascending=False)
    df_score = df_score.drop_duplicates()
    df_score = df_score[df_score['EstimateScore'] >= 9.5]
    df_rcm = pd.merge(df_score, df_info, on='Hotel ID', how='left')
    df_rcm = df_rcm.dropna(subset='Hotel_Name')
    df_rcm = df_rcm.drop(['Hotel_idx'], axis = 1).head()
    df_rcm

elif choice == 'Đề xuất dựa trên nội dung':    
    st.subheader("Content-Based")
    st.subheader("Hãy chọn 1 trong 2")
    status = st.radio("Tìm kiếm theo", ("Tên khách sạn", "Địa chỉ"))
    if status == 'Tên khách sạn':
        st.subheader("Tìm tên khách sạn")
        hotel_name = ''
        hotel_name = st.text_input("Nhập tên khách sạn cần tìm:")

        data = search_hotel_name(data_recommend, hotel_name)
        if hotel_name == '':
            st.session_state.data_recommend = data_recommend
        elif data.empty:
            st.write('Không có kết quả với từ khóa: ', '<' , hotel_name , '>')
            st.session_state.data_recommend = data
        else:
            st.write('Kết quả cho từ khóa  ' , '<' , hotel_name , '>')
            st.session_state.data_recommend = data
            st.write('Có ',len(data), ' kết quả tìm được')
    else:
        st.subheader("Tìm khách sạn theo địa chỉ")
        hotel_address = ''
        hotel_address = st.text_input("Nhập địa chỉ cần tìm:")

        data = search_hotel_address(data_recommend, hotel_address)
        if hotel_address == '':
            st.session_state.data_recommend = data_recommend
        elif data.empty:
            st.write('Không có kết quả với từ khóa: ', '<' , hotel_address , '>')
            st.session_state.data_recommend = data
        else:
            st.write('Kết quả cho từ khóa  ' , '<' , hotel_address , '>')
            st.session_state.data_recommend = data
            st.write('Có ',len(data), ' kết quả tìm được')

    ###### Giao diện Streamlit ######

    # Kiểm tra xem 'selected_hotel_id' đã có trong session_state hay chưa
    if 'selected_hotel_id' not in st.session_state:
        # Nếu chưa có, thiết lập giá trị mặc định là None hoặc ID khách sạn đầu tiên
        st.session_state.selected_hotel_id = None
    
    # Theo cách cho người dùng chọn khách sạn từ dropdown
    # Tạo một tuple cho mỗi khách sạn, trong đó phần tử đầu là tên và phần tử thứ hai là ID
    hotel_options = [(row['Hotel_Name'], row['Hotel_ID']) for index, row in st.session_state.data_recommend.iterrows()]
    st.session_state.data_recommend
    # Tạo một dropdown với options là các tuple này
    selected_hotel = st.selectbox(
        "Chọn khách sạn",
        options=hotel_options,
        format_func=lambda x: x[0]  # Hiển thị tên khách sạn
    )
    # Display the selected hotel
    st.write("Bạn đã chọn:", selected_hotel)

    st.subheader("Xác nhận chọn")
    submitted_rcm = st.button("Submit", key = 'btn2')
    if submitted_rcm:
        # Cập nhật session_state dựa trên lựa chọn hiện tại
        st.session_state.selected_hotel_id = selected_hotel[1]

        if st.session_state.selected_hotel_id:
            st.write("Hotel_ID: ", st.session_state.selected_hotel_id)
            # Hiển thị thông tin khách sạn được chọn
            selected_hotel = data_recommend[data_recommend['Hotel_ID'] == st.session_state.selected_hotel_id]

            if not selected_hotel.empty:
                st.write('#### Bạn vừa chọn:')
                st.write('### ', selected_hotel['Hotel_Name'].values[0])

                hotel_description = selected_hotel['Hotel_Description'].values[0]
                truncated_description = ' '.join(hotel_description.split()[:300])
                st.write('##### Thông tin:')
                st.write(truncated_description, '...')
                
                hotel_address = selected_hotel['Hotel_Address'].values[0]
                truncated_address = ' '.join(hotel_address.split())
                st.write('##### Địa chỉ:')
                st.write(truncated_address)

                st.write('##### Các khách sạn khác bạn cũng có thể quan tâm:')
                recommendations = get_recommendations(data_recommend, st.session_state.selected_hotel_id, cosine_sim=cosine_sim_new, nums=3) 
                display_recommended_hotels(recommendations, cols=3)
            else:
                st.write(f"Không tìm thấy khách sạn với ID: {st.session_state.selected_hotel_id}")
