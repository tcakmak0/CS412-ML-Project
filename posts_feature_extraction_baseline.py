# Initialize lists to store results
total_likes = []
total_comments = []
max_likes = []
captions_concatenated = []
averages = []


for r in range(0, 5415):
    total_like = 0
    total_comment = 0
    max_like = 0
    captions = ""
    for c in range(0, 35):
        post = posts.loc[r, c]
        # Check if post is not None
        if post is not None:
            like_count = post.get("like_count", 0)  
            comment_count = post.get("comments_count", 0)  

            # Calculate total likes and comments, and track the max likes
            total_like += int(like_count) if like_count is not None else 0
            total_comment += int(comment_count) if comment_count is not None else 0
            max_like = max(max_like, int(like_count) if like_count is not None else 0)
            captions += post.get("caption", "") if post.get("caption") is not None else ""
            captions += " "
            

    # Append the results for the current user (row)
    total_likes.append(total_like)
    total_comments.append(total_comment)
    max_likes.append(max_like)
    averages.append(total_like/35)  
    captions_concatenated.append(captions)

# After the loop, create a DataFrame to store these results
results_df = pd.DataFrame({
    'total_likes': total_likes,
    'total_comments': total_comments,
    'max_likes': max_likes,
    'average like': averages,
    'captions_concatenated': captions_concatenated
})

results_df.head()

results_df['captions_concatenated'] = results_df['captions_concatenated'].apply(preprocess_text).apply(remove_stopwords)