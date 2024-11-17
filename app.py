#Importing library
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from streamlit import pyplot
import preprocessor, helper
import nltk


#App title
st.sidebar.title("Whatsapp chat analyser")

#VADER : is a lexicon and rule-based sentiment analysis tool
nltk.download('vader_lexicon')

#file upload button
uploaded_file = st.sidebar.file_uploader("Choose a file")



if uploaded_file is not None:

    #getting byte form & then decoding
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")

    st.header("organised data") #subheading

    #perform preprocessing
    df = preprocessor.preprocessor(data)

    #show dataframe list
    st.dataframe(df)

    # fetch unique users
    user_list = df['user'].unique().tolist() #username list
    user_list.remove('WA_notification') #removeing system notifier
    user_list.sort() #sorting
    user_list.insert(0,'Overall') #insert 'Overall' at index 0

    # Select box
    selected_user = st.sidebar.selectbox("Show analysis wrt",user_list)

    ##############################################################################################

    # Importing SentimentIntensityAnalyser class from 'nltk.sentiment.vader'
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    # object
    sentiments = SentimentIntensityAnalyzer()

    # Creating different colums for (Positive/Negative/Neutral)
    df["po"] = [sentiments.polarity_scores(i)["pos"] for i in df["message"]]
    df["ne"] = [sentiments.polarity_scores(i)["neg"] for i in df["message"]]
    df["nu"] = [sentiments.polarity_scores(i)["neu"] for i in df["message"]]


    # To identify true sentiment per row in message column
    def sentiment(d):
        if d["po"] >= d["ne"] and d["po"] >= d["nu"]:
            return 1
        if d["ne"] >= d["po"] and d["ne"] >= d["nu"]:
            return -1
        if d["nu"] >= d["po"] and d["nu"] >= d["ne"]:
            return 0

    # Create new column in dataframe & applying function
    df['value'] = df.apply(lambda row: sentiment(row), axis=1)
    ##########################################################################################

    if st.sidebar.button("Show Analysis"):

        num_messages, num_words, num_media, num_links = helper.fetch_stats(selected_user,df)

        col1,col2,col3,col4 = st.columns(4)
        with col1:
            st.subheader("Total messages")
            st.title(num_messages)
        with col2:
            st.subheader("Total words")
            st.title(num_words)
        with col3:
            st.subheader("Total media file")
            st.title(num_media)
        with col4:
            st.subheader("Total links shared")
            st.title(num_links)

        # monthly timeline
        st.title("----------Timeline data----------")
        st.subheader("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user,df)

        fig, ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'], color='green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        #daily timeline
        st.subheader("Daily Timeline")
        timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(timeline['only_date'], timeline['message'], color='green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        #week activity map
        st.title('----------Activity map----------')
        col1, col2 = st.columns(2)

        with col1:
            st.header("Most busy day")
            busy_day = helper.week_activity_map(selected_user, df)
            fig, ax  = plt.subplots()
            ax.bar(busy_day.index, busy_day.values)
            pyplot(fig)

        with col2:
            st.header("Most busy month")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax  = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='orange')
            plt.xticks(rotation='vertical')
            pyplot(fig)

        #finding the busiest user in the group (not individual)
        if selected_user == 'Overall':
            st.header('Most busy users')
            x, new_df= helper.most_busy_users(df)
            fig, ax = plt.subplots()

            col1,col2 = st.columns(2)
            with col1:
                ax.bar(x.index, x.values, color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.subheader('Top user message (%)')
                st.dataframe(new_df)

        #wordCloud
        st.title("Wordcloud")
        df_wc = helper.create_wordcloud(selected_user,df)
        fig,ax = plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)

        # most common words
        st.header("Most common words")
        most_common_df = helper.most_common_words(selected_user,df)

        fig,ax = plt.subplots()
        ax.barh(most_common_df[0],most_common_df[1])
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # emoji analysis
        st.title('Emoji analysis')
        emoji_df = helper.emoji_helper(selected_user,df)
        col1,col2 = st.columns(2)
        with col1:
            st.dataframe(emoji_df)
        with col2:
            fig,ax = plt.subplots()
            ax.pie(emoji_df[1].head(), labels=emoji_df[0].head(),autopct="%0.2f")
            st.pyplot(fig)

        #Activity heatmap
        st.title('Weekly Activity Map')
        user_heatmap = helper.activity_heatmap(selected_user,df)
        fig,ax  = plt.subplots()
        ax = sns.heatmap(user_heatmap)
        st.pyplot(fig)

        # -------------------------Cumulative sentiment----------------------
        # Assuming 'message' column contains the text
        df['sentiment'] = df['message'].apply(lambda x: sentiments.polarity_scores(x)['compound'])

        df = df.sort_values('date').reset_index(drop=True)

        df['cumulative_sentiment'] = df['sentiment'].cumsum()

        plt.figure(figsize=(12, 6))
        plt.plot(df['date'], df['cumulative_sentiment'], color='blue', label='Cumulative Sentiment')
        plt.axhline(0, color='red', linestyle='--', linewidth=1, label='Neutral Line')
        plt.title('Cumulative Sentiment Over Time', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Cumulative Sentiment', fontsize=12)
        plt.legend()
        plt.grid(True)
        plt.show()

        # ------------------SENTIMENT ANALYSIS---------------------------#
        st.title("----------Sentiment Analysis----------")

        #---------------------Monthly activity map-------------------------
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<h3 style='text-align:center; color:black;'>Monthly Activity map(positive)</h3>",unsafe_allow_html=True)
            monthly_sentiment = helper.month_sentiment_map(selected_user, df, 1)
            fig, ax = plt.subplots()
            ax.bar(monthly_sentiment.index, monthly_sentiment.values, color='green')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.markdown("<h3 style='text-align: center; color: black;'>Monthly Activity map(Neutral)</h3>",unsafe_allow_html=True)

            monthly_sentiment = helper.month_sentiment_map(selected_user, df, 0)

            fig, ax = plt.subplots()
            ax.bar(monthly_sentiment.index, monthly_sentiment.values, color='grey')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col3:
            st.markdown("<h3 style='text-align: center; color: black;'>Monthly Activity map(Negative)</h3>",unsafe_allow_html=True)

            monthly_sentiment = helper.month_sentiment_map(selected_user, df, -1)

            fig, ax = plt.subplots()
            ax.bar(monthly_sentiment.index, monthly_sentiment.values, color='red')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        # ---------------------------Daily activity map-----------------------------------
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<h3 style='text-align: center; color: black;'>Daily Activity map(Positive)</h3>",unsafe_allow_html=True)

            daily_sentiment = helper.daily_sentiment_map(selected_user, df, 1)

            fig, ax = plt.subplots()
            ax.bar(daily_sentiment.index, daily_sentiment.values, color='green')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col2:
            st.markdown("<h3 style='text-align: center; color: black;'>Daily Activity map(Neutral)</h3>",unsafe_allow_html=True)

            daily_sentiment = helper.daily_sentiment_map(selected_user, df, 0)

            fig, ax = plt.subplots()
            ax.bar(daily_sentiment.index, daily_sentiment.values, color='grey')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col3:
            st.markdown("<h3 style='text-align: center; color: black;'>Daily Activity map(Negative)</h3>",unsafe_allow_html=True)

            daily_sentiment = helper.daily_sentiment_map(selected_user, df, -1)

            fig, ax = plt.subplots()
            ax.bar(daily_sentiment.index, daily_sentiment.values, color='red')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        # --------------------------Weekly heat map------------------------------
        col1, col2, col3 = st.columns(3)
        with col1:
            try:
                st.markdown("<h3 style='text-align: center; color: black;'>Weekly Activity Map(Positive)</h3>",unsafe_allow_html=True)

                weekly_heatmap = helper.sentiment_activity_heatmap(selected_user, df, 1)

                fig, ax = plt.subplots()
                ax = sns.heatmap(weekly_heatmap)
                st.pyplot(fig)
            except:
                st.image('error.webp')
        with col2:
            try:
                st.markdown("<h3 style='text-align: center; color: black;'>Weekly Activity Map(Neutral)</h3>",unsafe_allow_html=True)

                weekly_heatmap = helper.sentiment_activity_heatmap(selected_user, df, 0)

                fig, ax = plt.subplots()
                ax = sns.heatmap(weekly_heatmap)
                st.pyplot(fig)
            except:
                st.image('error.webp')
        with col3:
            try:
                st.markdown("<h3 style='text-align: center; color: black;'>Weekly Activity Map(Negative)</h3>",unsafe_allow_html=True)

                weekly_heatmap = helper.sentiment_activity_heatmap(selected_user, df, -1)

                fig, ax = plt.subplots()
                ax = sns.heatmap(weekly_heatmap)
                st.pyplot(fig)
            except:
                st.image('error.webp')

        #-------------------------Daily timeline------------------------------
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<h3 style='text-align: center; color: black;'>Daily Timeline(Positive)</h3>",unsafe_allow_html=True)

            daily_sentiment_timeline = helper.sentiment_daily_timeline(selected_user, df, 1)

            fig, ax = plt.subplots()
            ax.plot(daily_sentiment_timeline['only_date'], daily_sentiment_timeline['message'], color='green')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col2:
            st.markdown("<h3 style='text-align: center; color: black;'>Daily Timeline(Neutral)</h3>",unsafe_allow_html=True)

            daily_sentiment_timeline = helper.sentiment_daily_timeline(selected_user, df, 0)

            fig, ax = plt.subplots()
            ax.plot(daily_sentiment_timeline['only_date'], daily_sentiment_timeline['message'], color='grey')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col3:
            st.markdown("<h3 style='text-align: center; color: black;'>Daily Timeline(Negative)</h3>",unsafe_allow_html=True)

            daily_sentiment_timeline = helper.sentiment_daily_timeline(selected_user, df, -1)

            fig, ax = plt.subplots()
            ax.plot(daily_sentiment_timeline['only_date'], daily_sentiment_timeline['message'], color='red')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        #-----------------------------Monthly timeline---------------------------
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<h3 style='text-align: center; color: black;'>Monthly Timeline(Positive)</h3>",unsafe_allow_html=True)

            timeline = helper.sentiment_monthly_timeline(selected_user, df, 1)

            fig, ax = plt.subplots()
            ax.plot(timeline['time'], timeline['message'], color='green')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col2:
            st.markdown("<h3 style='text-align: center; color: black;'>Monthly Timeline(Neutral)</h3>",unsafe_allow_html=True)

            timeline = helper.sentiment_monthly_timeline(selected_user, df, 0)

            fig, ax = plt.subplots()
            ax.plot(timeline['time'], timeline['message'], color='grey')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col3:
            st.markdown("<h3 style='text-align: center; color: black;'>Monthly Timeline(Negative)</h3>",unsafe_allow_html=True)

            timeline = helper.sentiment_monthly_timeline(selected_user, df, -1)

            fig, ax = plt.subplots()
            ax.plot(timeline['time'], timeline['message'], color='red')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)


        #------------------------Percentage contributed---------------------------------------
        if selected_user == 'Overall':
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("<h3 style='text-align: center; color: black;'>Most Positive Contribution</h3>",
                            unsafe_allow_html=True)
                x = helper.percentage(df, 1)

                # Displaying
                st.dataframe(x)
            with col2:
                st.markdown("<h3 style='text-align: center; color: black;'>Most Neutral Contribution</h3>",
                            unsafe_allow_html=True)
                y = helper.percentage(df, 0)

                # Displaying
                st.dataframe(y)
            with col3:
                st.markdown("<h3 style='text-align: center; color: black;'>Most Negative Contribution</h3>",
                            unsafe_allow_html=True)
                z = helper.percentage(df, -1)

                # Displaying
                st.dataframe(z)


        #-------------------Most Positive,Negative,Neutral User------------------------
        if selected_user == 'Overall':
            # Getting names per sentiment
            x = df['user'][df['value'] == 1].value_counts().head(10)
            y = df['user'][df['value'] == -1].value_counts().head(10)
            z = df['user'][df['value'] == 0].value_counts().head(10)

            col1, col2, col3 = st.columns(3)
            with col1:
                # heading
                st.markdown("<h3 style='text-align: center; color: black;'>Most Positive Users</h3>",unsafe_allow_html=True)

                # Displaying
                fig, ax = plt.subplots()
                ax.bar(x.index, x.values, color='green')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                # heading
                st.markdown("<h3 style='text-align: center; color: black;'>Most Neutral Users</h3>",unsafe_allow_html=True)

                # Displaying
                fig, ax = plt.subplots()
                ax.bar(z.index, z.values, color='grey')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col3:
                # heading
                st.markdown("<h3 style='text-align: center; color: black;'>Most Negative Users</h3>",unsafe_allow_html=True)

                # Displaying
                fig, ax = plt.subplots()
                ax.bar(y.index, y.values, color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

        #----------------------------WORDCLOUD-------------------------------------
        col1, col2, col3 = st.columns(3)
        with col1:
            try:
                # heading
                st.markdown("<h3 style='text-align: center; color: black;'>Positive WordCloud</h3>",unsafe_allow_html=True)

                # Creating wordcloud of positive words
                df_wc = helper.sentiment_wordcloud(selected_user, df, 1)
                fig, ax = plt.subplots()
                ax.imshow(df_wc)
                st.pyplot(fig)
            except:
                # Display error message
                st.image('error.webp')
        with col2:
            try:
                # heading
                st.markdown("<h3 style='text-align: center; color: black;'>Neutral WordCloud</h3>",unsafe_allow_html=True)

                # Creating wordcloud of neutral words
                df_wc = helper.sentiment_wordcloud(selected_user, df, 0)
                fig, ax = plt.subplots()
                ax.imshow(df_wc)
                st.pyplot(fig)
            except:
                # Display error message
                st.image('error.webp')
        with col3:
            try:
                # heading
                st.markdown("<h3 style='text-align: center; color: black;'>Negative WordCloud</h3>",unsafe_allow_html=True)

                # Creating wordcloud of negative words
                df_wc = helper.sentiment_wordcloud(selected_user, df, -1)
                fig, ax = plt.subplots()
                ax.imshow(df_wc)
                st.pyplot(fig)
            except:
                # Display error message
                st.image('error.webp')

        #------------------------Most common positive words-------------------------
        col1, col2, col3 = st.columns(3)
        with col1:
            try:
                # Data frame of most common positive words.
                most_common_df = helper.sentiment_common_words(selected_user, df, 1)

                # heading
                st.markdown("<h3 style='text-align: center; color: black;'>Positive Words</h3>", unsafe_allow_html=True)
                fig, ax = plt.subplots()
                ax.barh(most_common_df[0], most_common_df[1], color='green')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            except:
                # Disply error image
                st.image('error.webp')
        with col2:
            try:
                # Data frame of most common neutral words.
                most_common_df = helper.sentiment_common_words(selected_user, df, 0)

                # heading
                st.markdown("<h3 style='text-align: center; color: black;'>Neutral Words</h3>", unsafe_allow_html=True)
                fig, ax = plt.subplots()
                ax.barh(most_common_df[0], most_common_df[1], color='grey')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            except:
                # Disply error image
                st.image('error.webp')
        with col3:
            try:
                # Data frame of most common negative words.
                most_common_df = helper.sentiment_common_words(selected_user, df, -1)

                # heading
                st.markdown("<h3 style='text-align: center; color: black;'>Negative Words</h3>", unsafe_allow_html=True)
                fig, ax = plt.subplots()
                ax.barh(most_common_df[0], most_common_df[1], color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            except:
                # Disply error image
                st.image('error.webp')





