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

class LogousChatbotScenarioBase(object):
    
    def parse_yes_no(self, sentence, ner_results):
        try:
            sentence = sentence.lower().strip()
            if (("evet" in sentence) or ("uygun" in sentence) or ("tamam" in sentence)):
                result = True
            else:
                result =  False
                
            error = ScenarioUtils.CONTINUE_SCENARIO
        except:
            result = False
            error = ScenarioUtils.PARSE_ERROR
        #print("parse_yes_no---------------->" + str(result))
                    
        return result, error
        
    def parse_day(self, sentence, ner_results):        
        error = False
        result = -1 
        try:            
            for named_entity_cls in ner_results:                
                if (named_entity_cls == "DATE"):
                    for ne in ner_results[named_entity_cls]:                        
                        nes = ne[0].split()                                         
                        result = int(nes[0])                        
                        if ("hafta" in nes[1]):
                            result = result * 7                            
                        elif ("ay" in nes[1]):
                            result = result * 30
                        return result, ScenarioUtils.CONTINUE_SCENARIO
                    
                elif (named_entity_cls == "CARDINAL"):
                    for ne in ner_results[named_entity_cls]:                        
                        result = int(ne[0])
                        return result, ScenarioUtils.CONTINUE_SCENARIO                                    
        except:
            error = ScenarioUtils.PARSE_ERROR
            result = -1        
        return result, error
        
    def parse_date(self, sentence, ner_results):    
        result = None
        try:
            import re, dateparser            
            x = re.findall("\s*(3[01]|[12][0-9]|0?[1-9])(\.|\/)(1[012]|0?[1-9])(\.|\/)((?:19|20)\d{2})\s*", sentence)  
            
            current = "".join(x[0])            
            if (len(x)>0):
                result = dateparser.parse(current)
                            
            if (result == None):
                for named_entity_cls in ner_results:
                    if (named_entity_cls == "DATE"):
                        result = dateparser.parse(ne[0])
                        break
                        
            if (result == None):
                error = ScenarioUtils.PARSE_ERROR
            else:
                result = str(result)
                error = ScenarioUtils.CONTINUE_SCENARIO
        except:
            error = ScenarioUtils.PARSE_ERROR
            result = None
        #print("parse_date---------------->" + str(result))
        return result, error