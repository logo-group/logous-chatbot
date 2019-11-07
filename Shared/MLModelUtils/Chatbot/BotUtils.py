import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from TurkishStemmer import TurkishStemmer
import heapq 
import pandas as pd
import warnings
from Shared.MLModelUtils.Chatbot.ScenarioUtils import ScenarioUtils

class BotUtils(object):
    
    def __init__(self, scenarios, global_context):
        self.scenarios = scenarios
        self.global_context = global_context
    
    def clean_alpha_num(self, w):
        non_alpha_cleaned = ""
        for c in w:
            if c.isalpha():
                non_alpha_cleaned = non_alpha_cleaned + c

        return non_alpha_cleaned.strip()

    # Preprocess sentence
    def preprocess(self, dt):

        lower_map = {
            ord(u'I'): u'ı',
            ord(u'İ'): u'i',
        }

        dt = dt.translate(lower_map).lower()     
        dt = dt.replace("'"," ")
        tokens = word_tokenize(dt)    
        # remove all tokens that are not alphabetic
        words = [self.clean_alpha_num(word) for word in tokens if len(self.clean_alpha_num(word))>0]
        #print(words)
        # remove stopwords
        from nltk.corpus import stopwords
        stop_words = stopwords.words('turkish')
        stop_words = np.concatenate((stop_words,stopwords.words('english')))
        stop_words2 = ["acaba","adeta","ait","altı","ama","ancak","artık","aslında","asıl","ayrıca","bazen","başka","belki","ben","beri","beş","bide","bir","biraz","birkaç","birçok","biz","bu","bura","böyle","cuma","cumartesi","da","dahil","dair","de","defa","diye","diğer","dokuz","dolayı","dört","en","et","eğer","fakat","falan","filan","galiba","gel","gene","gibi","göre","hadi","hangi","hem","herhalde","herhangi","iki","ile","için","kere","kez","kim","kimi","lakin","lütfen","mesela","mi","mü","mı","new","niye","ol","on","oysa","pazar","pazartesi","pek","perşembe","rağmen","resmen","salı","sekiz","sen","seni","siz","sırf","tabi","tabii","tane","the","un","vala","var","veya","yada","yahu","yaklaşık","yani","yap","yedi","yoksa","zaten","çarşamba","çok","çünkü","üzere","üç","şey","şu"]
        stop_words =  np.unique(np.concatenate((stop_words,stop_words2)))

        words = [word for word in words if word not in stop_words and len(word)>1]

        # Stem words
        porter = TurkishStemmer()
        words = [porter.stem(word) for word in words]

        return words    

    # calculate sentence signature
    def calculate_sentence_signature(self, sentence, model):
        sentenceWords = self.preprocess(sentence)
        sentenceWVs = []
        for w in sentenceWords:
            try:
                sentenceWVs.append(model[w])
            except:
                pass

        if (len(sentenceWVs)==0):
            return []
        
        sentenceWVs = np.array(sentenceWVs)
        signature = np.mean(sentenceWVs, axis=0)
        return signature

    # Calculate answer signatures
    def calculate_answer_hashes(self, questions, answers, model):
        signatures = []
        
        for q in questions:
            signature = self.calculate_sentence_signature(q, model)
            signatures.append(signature.reshape(1,len(signature)))            

        signatures = np.array(signatures)
        return signatures, answers


    # find answer for given question sentence (LSH is required)
    def find_answer(self, question, questionSignature, signatures, answers, questions, model, heapsize=5):
        heaplist = []    
        questionSignature = questionSignature.reshape(1,len(questionSignature))
        
        for i in range(0,np.shape(signatures)[0]):
                        
            similarity = cosine_similarity(questionSignature, signatures[i])

            heapq.heappush(heaplist, (similarity, answers[i], questions[i]))

            if (len(heaplist)>heapsize):
                heapq.heappop(heaplist)                

        ans_list = heapq.nlargest(heapsize, heaplist)    
        #answer = (ans_list[0])[1]
        
        ######## ngram sim ranking for w2v ensemble results            
        import nltk       
        maxval = -1
        maxelement = None
        n = 3
        ngram_set_1 = set(nltk.ngrams(' '.join(self.preprocess(question)), n=n))          
        for current in ans_list:             
            question_of_current_answer = current[2]              
            ngram_set_2 =set(nltk.ngrams(' '.join(self.preprocess(question_of_current_answer)), n=n))                
            sim = self.calculate_ngram_sim(ngram_set_1, ngram_set_2)     
            if (sim > maxval):
                maxval = sim
                maxelement = current

        #result = {"results": maxelement[1]}
        answer = maxelement[1]       
        
                                        
        scenarioUtils = ScenarioUtils()        
        result = { "answer": answer }   
        
        if (ScenarioUtils.RUN_SCENARIO in answer):
            target_scenario_id = int(answer.split(':')[1])
            context, command, answer = scenarioUtils.run_scenario(self.get_scenarios(), target_scenario_id, answer, self.global_context)
            result["answer"] = answer
            if ((command == ScenarioUtils.CONTINUE_SCENARIO) or (command == ScenarioUtils.PARSE_ERROR)):
                result["context"] = context
            else:
                result["context"] = None
                        
        return result
    
    def get_scenarios(self):
        return self.scenarios  

    def process_msg(self, data, signatures, answers, questions, embedding_model):
        if ("context" in data) and (data["context"] != None) and ("scenario_id" in data["context"]):
            scenarioUtils = ScenarioUtils()
            context, command, answer = scenarioUtils.run_scenario(self.get_scenarios(), data["context"]["scenario_id"], data["question"], self.global_context, data["context"])
            result = {}
            result["answer"] = answer
            print(answer)
            if ((command == ScenarioUtils.CONTINUE_SCENARIO) or (command == ScenarioUtils.PARSE_ERROR)):
                result["context"] = context
            else:
                result["context"] = None   
            result = {"results": result}        
        else:        
            questionSignature = self.calculate_sentence_signature(data["question"], embedding_model)                                             
            if len(questionSignature)==0:
                result = {"results": "Üzgünüm uygun bir cevap bulamadım."}
            else:
                ans = self.find_answer(data["question"], questionSignature, signatures, answers, questions, embedding_model, heapsize=10)                                              
                result = {"results": ans}
        return result

    def calculate_ngram_sim(self, n1, n2):        
        total = 0
        if (len(n1)<len(n2)):
            for i in n1:
                if i in n2:
                    total = total + 1
        else:
            for i in n2:
                if i in n1:
                    total = total + 1

        return total

    """
    Read the file given as parameter
    """
    def read_file(self, filename):
        f = open(filename, "r", encoding="utf-8")
        file_text = f.read()
        f.close()
        return file_text

    """
    Write to the file given as parameter
    """
    def write_file(self, filename, output):
        full_filename = './' + filename + '.txt'
        file = open(full_filename, 'w', encoding = "utf-8")
        file.write(str(output))
        file.close()

    """
    Read the pickle file given as parameter
    """
    def open_pickle(self, filename):
        infile = open(filename,'rb')
        opened_pickle = pickle.load(infile)
        infile.close()
        return opened_pickle

    """
    Update the frequency dictionary for given word
    """
    def add_to_freq_dict(self, dictionary, word):
        if word not in dictionary:
            freq = 1
            dictionary[word] = freq
        else:
            dictionary[word] += 1
    """
    Delete the characters in the 'char_list' from the text
    """
    def delete_characters(self, text, char_list):
        for char in char_list:
            text = re.sub(char, '', text)
        return text

    def delete_characters_space(self, text, char_list):
        for char in char_list:
            text = re.sub(char, ' ', text)
        return text

    """
    Remove the words in the 'delete_list'
    """
    def remove_words(self, word_list, delete_list):
        return [word for word in word_list if word not in delete_list]

    """
    Remove the words if they contain certain patterns
    """
    def remove_with_regex(self, word):
        check = re.findall(r'(?:pic.twitter|^@|^rt$)', word)
        if check:
            word = ""
        return(word)

    """
    Remove digits
    """
    def remove_decimal(word):
        new_word_list = []
        for word in word_list:
            check = re.findall(r'(?:\d+)', word)
            if not check:
                new_word_list.append(word)
        return new_word_list

    """
    Replace similar emoticons with a determined keyword
    """
    def replace_emoticon(word):
        check_pos = re.findall(r'(?::\)|:-\)|=\)|:D|:d|<3|\(:|:\'\)|\^\^|;\)|\(-:)', word)
        check_neg = re.findall(r'(:-\(|:\(|;\(|;-\(|=\(|:/|:\\|-_-|\):|\)-:)', word)
        if check_pos:
            #word = ":)"
            word = "SMILEYPOSITIVE"
        elif check_neg:
            #word = ":("
            word = "SMILEYNEGATIVE"
        return word

    """
    Remove punctuations from the word
    """
    def remove_punct(word):
        exclude = set(string.punctuation)
        word = replace_emoticon(word)
        word = ''.join(ch for ch in word if ch not in exclude)
        return word

    """
    Replace some turkish characters with their corresponding english letters
    """
    def replace_turkish_char(word):
        word = word.replace('ş','s')
        word = word.replace('ç','c')
        word = word.replace('ğ','g')
        word = word.replace('ü','u')
        word = word.replace('ö','o')
        word = word.replace('ı','i')
        return word

    """
    Remove letters if they repeat consecutively to one letter 
    """
    def remove_repeating_char(word):
        new_word = ""
        prev_char = ''
        for char in word:
            if prev_char == char:
                continue
            new_word = new_word + char
            prev_char = char
        return new_word