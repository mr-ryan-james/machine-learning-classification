import graphlab

def remove_punctuation(text):
    import string
    return text.translate(None, string.punctuation) 

products = graphlab.SFrame('amazon_baby.gl/')

products['review_clean'] = products['review'].apply(remove_punctuation)
products = products.fillna('review', '')  # fill in N/A's in the review column
products = products[products['rating'] != 3]
products['sentiment'] = products['rating'].apply(lambda rating : +1 if rating > 3 else -1)
products['word_count'] = graphlab.text_analytics.count_words(products['review_clean'])

train_data, test_data = products.random_split(.8, seed=1)

sentiment_model = graphlab.logistic_classifier.create(train_data,
                                                      target = 'sentiment',
                                                      features=['word_count'],
                                                      validation_set=None)

weights = sentiment_model.coefficients
print weights.column_names()
print weights

num_positive_weights = len(weights[weights["value"] > 0])
num_negative_weights = len(weights[weights["value"] < 0])

print "QUIZ QUESTION --> Number of positive weights: %s " % num_positive_weights
print "Number of negative weights: %s " % num_negative_weights
print "Total: %s " % len(weights)

#Making Predictions with logistic regression

sample_test_data = test_data[10:13]
print sample_test_data['rating']
sample_test_data.print_rows()

print sample_test_data[0]['review']
print sample_test_data[1]['review']

scores = sentiment_model.predict(sample_test_data, output_type='margin')
print "Scores: %s " % scores

def predictSentiment(test_data):
    return sentiment_model.predict(test_data)

print predictSentiment(sample_test_data)

print "Class predictions according to GraphLab Create:" 
print sentiment_model.predict(sample_test_data, output_type='probability')
print "QUIZ QUESTION --> THIRD ONE"

test_data["score"] = sentiment_model.predict(test_data, output_type='probability')
print "top twenty positive"
print test_data.topk("score", k=20).print_rows(num_rows=20)
print "top twenty negative"
print test_data.topk("score", k=20, reverse=True).print_rows(num_rows=20)

def get_classification_accuracy(model, data, true_labels):
    # First get the predictions
    ## YOUR CODE HERE
    predictions = model.predict(data)
    
    # Compute the number of correctly classified examples
    ## YOUR CODE HERE
    num_objectively_correct = 0

    for x in range(0,len(data)):
        if predictions[x] == true_labels[x]:
            num_objectively_correct += 1



    # Then compute accuracy by dividing num_correct by total number of examples
    ## YOUR CODE HERE
    accuracy = num_objectively_correct / float(len(data))
    
    return accuracy

print "Accuracy --> 0.914536837053" # % get_classification_accuracy(sentiment_model, test_data, test_data['sentiment'])


significant_words = ['love', 'great', 'easy', 'old', 'little', 'perfect', 'loves', 
      'well', 'able', 'car', 'broke', 'less', 'even', 'waste', 'disappointed', 
      'work', 'product', 'money', 'would', 'return']

print len(significant_words)

train_data['word_count_subset'] = train_data['word_count'].dict_trim_by_keys(significant_words, exclude=False)
test_data['word_count_subset'] = test_data['word_count'].dict_trim_by_keys(significant_words, exclude=False)

print train_data[0]['review']

print train_data[0]['word_count']

print train_data[0]['word_count_subset']


simple_model = graphlab.logistic_classifier.create(train_data,
                                                   target = 'sentiment',
                                                   features=['word_count_subset'],
                                                   validation_set=None)
print simple_model

#print get_classification_accuracy(simple_model, test_data, test_data['sentiment'])

print simple_model.coefficients

print simple_model.coefficients.sort('value', ascending=False).print_rows(num_rows=21)
print sentiment_model.coefficients.sort('value', ascending=False).print_rows(num_rows=200)

print "which one is better?"
#print "sentiment_model --> train -> %s" % get_classification_accuracy(sentiment_model, train_data, train_data['sentiment']) # 0.979
#print "simple_model --> train -> %s" %  get_classification_accuracy(simple_model, train_data, train_data['sentiment']) # 0.8668

#print "sentiment_model --> test -> %s" % get_classification_accuracy(sentiment_model, test_data, test_data['sentiment']) # 0.9145
#print "simple_model --> test -> %s" %  get_classification_accuracy(simple_model, test_data, test_data['sentiment']) # 0.869


print "Majority class predictions"

num_positive  = (train_data['sentiment'] == +1).sum()
num_negative = (train_data['sentiment'] == -1).sum()
print "train data"
print num_positive
print num_negative

print "test data"
num_positive  = (test_data['sentiment'] == +1).sum()
num_negative = (test_data['sentiment'] == -1).sum()
print num_positive
print num_negative
