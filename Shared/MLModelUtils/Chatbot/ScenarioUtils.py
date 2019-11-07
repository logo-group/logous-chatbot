import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from TurkishStemmer import TurkishStemmer
import heapq 
import pandas as pd
import warnings

class ScenarioUtils(object):

    RUN_SCENARIO = "$run_scenario"
    ABORT_SCENARIO = "$abort_scenario"
    CONTINUE_SCENARIO = "$continue_scenario"
    FINISH_SCENARIO = "$finish_scenario"
    PARSE_ERROR = "$parse_error"
    
    def run_scenario(self, sclist, scenario_id, sentence, global_context, context = None):

        sce = sclist[scenario_id]        
        bot = global_context[sce["bot_main_class"]]
        
        if (context == None):
            context = { "scenario_id" : scenario_id, "current_entity" : None, "entities" : {}}        

        # Process abort related command from user
        if (self.check_for_abort(sentence)):
            return context, ScenarioUtils.ABORT_SCENARIO, sce["abort_msg"]

        # Process user message
        
        # Perform NER
        if (sce["ml_api_url"] is not None):
            import requests, json
            data = {
              "force_to_fetch_model": False,
              "modelid": sce["ner_model_id"],      
              "texts": sentence
            }
            response = requests.post(sce["ml_api_url"], json=data)
            ner_results = json.loads(response.text)
            ner_results = ner_results["results"]
        else:
            ner_results = []
        
        # Parse the user input for current entity
        if (context["current_entity"]):
            result, command = getattr(bot, 'parse_entity_%s' % context["current_entity"])(bot, context, sentence, ner_results)
            
            # In case of any abort command is received from entity parser, abort whole scenario            
            if (command == ScenarioUtils.ABORT_SCENARIO):
                return context, ScenarioUtils.ABORT_SCENARIO, sce["abort_msg"]
                
            if (command == ScenarioUtils.CONTINUE_SCENARIO):
                context["entities"][context["current_entity"]] = result
            else:                
                return context, ScenarioUtils.PARSE_ERROR, sce["error_msg"]    
                    
        for e in sce["flow"]:
            if (e not in context["entities"]):   

                if ("msg" in sce["entities"][e]):
                    question = self.enrich_msg(sce, sce["entities"][e]["msg"], context)
                    operation = ScenarioUtils.CONTINUE_SCENARIO
                else:
                    question, operation = self.enrich_msg(sce, getattr(bot, 'ask_entity_%s' % e)(bot, context, sentence, ner_results),context)

                context["current_entity"] = e                    
                return context, operation, question
        
        return context, ScenarioUtils.FINISH_SCENARIO, sce["final_msg"]

    def check_for_abort(self, sentence):
        sentence = sentence.lower().strip()
        if ((sentence == "iptal") or (sentence == "vazgeç") or (sentence == "iptal et")):
            return True
        else:
            return False

    def enrich_msg(self, sce, msg, context):
                
        for e in sce["entities"]:               
            if (e in context["entities"]):
                msg = msg.replace("#" + e, str(context["entities"][e]))

        return msg