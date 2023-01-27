import json
import os
import pandas as pd
import numpy as np

import sparknlp_jsl
import sparknlp
from pyspark.ml import Pipeline,PipelineModel
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import monotonically_increasing_id

from sparknlp.annotator import *
from sparknlp_jsl.annotator import *
from sparknlp.base import *
from sparknlp_jsl.structured_deidentification import StructuredDeidentification
from sparknlp.pretrained import PretrainedPipeline

class Deidentifier():

    def __init__(self, spark):
        self.spark = spark

        """ This class is used to deidentify the given data. 

        Parameters
        ----------
        spark : SparkSession
            SparkSession object
        
        """


    def deid_with_custom_pipeline(self):

        """ This function is used to deidentify the given data with custom pipeline."""

        #---In case of mode is "mask"---

        if self.mode=="mask" and self.unnormalized_date==False:

            if self.masking_policy=="entity_labels":
                #deid model with "entity_labels"
                deidentification= DeIdentification()\
                    .setInputCols([self.sentence, self.token, self.ner_chunk])\
                    .setOutputCol("deidentified")\
                    .setMode(self.mode)\
                    .setMaskingPolicy(self.masking_policy)\
                    .setOutputAsDocument(True)
            
            elif self.masking_policy=="same_length_chars":
                #deid model with "same_length_chars"
                deidentification= DeIdentification()\
                    .setInputCols([self.sentence, self.token, self.ner_chunk])\
                    .setOutputCol("deidentified")\
                    .setMode(self.mode)\
                    .setMaskingPolicy(self.masking_policy)\
                    .setOutputAsDocument(True)
            
            elif self.masking_policy=="fixed_length_chars":
                #deid model with "fixed_length_chars"
                deidentification= DeIdentification()\
                    .setInputCols([self.sentence, self.token, self.ner_chunk])\
                    .setOutputCol("deidentified")\
                    .setMode(self.mode)\
                    .setMaskingPolicy(self.masking_policy)\
                    .setFixedMaskLength(self.fixed_mask_length)\
                    .setOutputAsDocument(True)

        
        #---In case of mode is "obfuscate"---

        elif self.mode=="obfuscate" and self.unnormalized_date==False and self.shift_days==False:

            if self.obfuscate_ref_source=="faker":     
                deidentification = DeIdentification()\
                        .setInputCols([self.sentence, self.token, self.ner_chunk]) \
                        .setOutputCol("deidentified") \
                        .setMode(self.mode)\
                        .setObfuscateDate(self.obfuscate_date)\
                        .setObfuscateRefSource(self.obfuscate_ref_source)\
                        .setOutputAsDocument(True)
        
            elif self.obfuscate_ref_source=="both" or self.obfuscate_ref_source=="file" :     
                deidentification = DeIdentification()\
                        .setInputCols([self.sentence, self.token, self.ner_chunk]) \
                        .setOutputCol("deidentified") \
                        .setMode(self.mode)\
                        .setObfuscateDate(self.obfuscate_date)\
                        .setObfuscateRefFile(self.obfuscate_ref_file_path)\
                        .setObfuscateRefSource(self.obfuscate_ref_source) \
                        .setOutputAsDocument(True)
        
            elif self.age_group_obfuscation==True:
                deidentification = DeIdentification()\
                        .setInputCols([self.sentence, self.token, self.ner_chunk]) \
                        .setOutputCol("deidentified") \
                        .setMode(self.mode)\
                        .setObfuscateDate(self.obfuscate_date)\
                        .setObfuscateRefSource("faker") \
                        .setAgeRanges(self.age_ranges)\
                        .setOutputAsDocument(True)

        #--------Shifting days according to the ID column------------
        #--------DocumentHashCoder Should Be Feed By the USER-----

        elif self.shift_days==True and self.mode=="obfuscate" and self.unnormalized_date==False:
                deidentification = DeIdentification()\
                            .setInputCols([self.documentHashCoder_col_name, self.token, self.ner_chunk])\
                            .setOutputCol("deidentified") \
                            .setDateTag("DATE") \
                            .setMode(self.mode)\
                            .setObfuscateDate(self.obfuscate_date)\
                            .setObfuscateRefSource("faker") \
                            .setLanguage(self.language)\
                            .setRegion(self.region)\
                            .setUseShifDays(self.shift_days)\
                            .setOutputAsDocument(True)


        #--------Unnormalized Date------------

        elif self.unnormalized_date==True:
                if self.unnormalized_mode=="mask":

                    deidentification = DeIdentification() \
                            .setInputCols([self.ner_chunk, self.token, self.documentHashCoder_col_name]) \
                            .setOutputCol("deidentified") \
                            .setMode(self.mode) \
                            .setObfuscateDate(self.obfuscate_date) \
                            .setDateTag(self.date_tag) \
                            .setLanguage(self.language) \
                            .setObfuscateRefSource('faker') \
                            .setUseShifDays(True)\
                            .setRegion(self.region)\
                            .setUnnormalizedDateMode(self.unnormalized_mode)\
                            .setOutputAsDocument(True)


                elif self.unnormalized_mode=="obfuscate":

                        deidentification = DeIdentification() \
                            .setInputCols([self.ner_chunk, self.token, self.documentHashCoder_col_name]) \
                            .setOutputCol('deidentified') \
                            .setMode(self.unnormalized_mode) \
                            .setObfuscateDate(self.obfuscate_date) \
                            .setDateTag(self.date_tag) \
                            .setLanguage(self.language) \
                            .setObfuscateRefSource('faker') \
                            .setUseShifDays(True)\
                            .setRegion(self.region)\
                            .setUnnormalizedDateMode(self.unnormalized_mode)\
                            .setOutputAsDocument(True)
          

        try:
            input_file_type= self.input_file_path.split(".")[-1]
        
            if input_file_type=="csv":
                data = self.spark.read.csv(self.input_file_path, header=True, sep=self.separator)

            elif input_file_type=="json":
                data= self.spark.read.format(input_file_type).load(self.input_file_path)
        
        except:
            raise Exception("You entered an invalid file path or file format...")
     

        
        df = pd.DataFrame(columns=["ID"])

        try:
            for i in self.field_names:
                print(f"Deidentification process of '{i}' field is starting...")
                if i != "text":
                    self.custom_pipeline.stages[0].setInputCol(f"{i}")
                    
                elif i=="text":
                    self.custom_pipeline.stages[0].setInputCol(f"{i}")
                
                deid_pipeline= Pipeline(stages=[
                    self.custom_pipeline,
                    deidentification])  
                    
                model= deid_pipeline.fit(data)
                result= model.transform(data)
                
                result = result.withColumn("ID", monotonically_increasing_id()) 
                output_df= result.select("ID", F.explode(F.arrays_zip("document.result", "deidentified.result")).alias("cols")) \
                                .select("ID",
                                        F.expr("cols['0']").alias(f"{i}"), 
                                        F.expr("cols['1']").alias(f"{i}_deidentified")).toPandas()

                df = pd.merge(df, output_df, on='ID', how='outer')
                df.reset_index(drop=True, inplace=True)

                print(f"Deidentification process of '{i}' field is completed...")
        except:
            raise Exception("You entered either an invalid field name or wrong separator for input csv file...")
            


        if input_file_type=="csv":
            df.to_csv(f"{self.output_file_path}", index=False)
            print(f"Deidentifcation successfully completed and the results saved as '{self.output_file_path}' !")
            
        
        elif input_file_type=="json":
            df.to_json(f"{self.output_file_path}", orient="records")
            print(f"Deidentifcation successfully completed and the results saved as '{self.output_file_path}' !")


        df= df.fillna("NONE")
        return_df= self.spark.createDataFrame(df)    

        return return_df




### ---------------------Deidentification with pretrained pipeline--------------------- ###


    def deid_with_pp(self):

        """Deidentification with pretrained pipeline"""

        deid_pp = PretrainedPipeline("clinical_deidentification", "en", "clinical/models")
        
        #setting the output as document format
        for i in np.arange(-5,-2,1):
            deid_pp.model.stages[i].setOutputAsDocument(True)

        
        if self.mode=="mask" and self.unnormalized_date==False:

            if self.masking_policy=="fixed_length_chars":
                deid_pp.model.stages[-3].setFixedMaskLength(self.fixed_mask_length)

        elif self.mode=="obfuscate" and self.unnormalized_date==False:

            deid_pp.model.stages[-2].setObfuscateDate(self.obfuscate_date)
            deid_pp.model.stages[-2].setObfuscateRefSource(self.obfuscate_ref_source)

            
            #--------Age group obfuscation------------

            if self.age_group_obfuscation==True:
                deid_pp.model.stages[-2].setObfuscateDate(self.obfuscate_date)
                deid_pp.model.stages[-2].setObfuscateRefSource("faker")
                deid_pp.model.stages[-2].setAgeRanges(self.age_ranges) 

            
            #--------Shifting days------------

            elif self.shift_days==True and self.number_of_days is not None:
                deid_pp.model.stages[-2].setObfuscateDate(self.obfuscate_date)
                deid_pp.model.stages[-2].setDateTag(self.date_tag)
                deid_pp.model.stages[-2].setLanguage(self.language)
                deid_pp.model.stages[-2].setObfuscateRefSource('faker')
                deid_pp.model.stages[-2].setUseShifDays(True)
                deid_pp.model.stages[-2].setRegion(self.region)
                deid_pp.model.stages[-2].setDays(self.number_of_days)
        
 

        
        #--------Masking Unnormalized Date Formats------------

        elif self.unnormalized_date==True:
            deid_pp.model.stages[-2].setObfuscateDate(self.obfuscate_date)
            deid_pp.model.stages[-2].setDateTag(self.date_tag)
            deid_pp.model.stages[-2].setLanguage(self.language)
            deid_pp.model.stages[-2].setObfuscateRefSource('faker')
            deid_pp.model.stages[-2].setUseShifDays(True)
            deid_pp.model.stages[-2].setRegion(self.region)
            deid_pp.model.stages[-2].setUnnormalizedDateMode(self.unnormalized_mode)


        try:
            input_file_type= self.input_file_path.split(".")[-1]
            if input_file_type=="csv":
                data = self.spark.read.csv(self.input_file_path, header=True, sep=self.separator)

            elif input_file_type=="json":
                data= self.spark.read.format(input_file_type).load(self.input_file_path)
        
        except:   
            raise Exception("You entered an invalid file path or file format...")
            


        df = pd.DataFrame(columns=["ID"])

        try:
            for i in self.field_names:
                print(f"Deidentification process of '{i}' field is starting...")

                if i != "text":
                    df_deid=  data.select(f"{i}")
                    df_deid= df_deid.withColumnRenamed(f"{i}", "text")
                    
                elif i=="text":
                    df_deid=  data.select(f"{i}")
                    df_deid= data

                                        
                #deid_res= deid_pp.transform(df_deid.select("text"))
                deid_res= deid_pp.transform(df_deid)
                
                deid_res= deid_res.withColumn("ID", monotonically_increasing_id())
                

                try:
                    if self.mode=="mask" and self.masking_policy=="entity_labels":
                        pd_deid= deid_res.select("ID", "text", "finished_masked").toPandas()
                        pd_deid= pd_deid.rename(columns={"finished_masked": f"{i}_deid", 
                                                        "text": f"{i}_original"})

                    elif self.mode=="mask" and self.masking_policy=="same_length_chars":
                        pd_deid= deid_res.select("ID", "text", "finished_masked_with_chars").toPandas()
                        pd_deid= pd_deid.rename(columns={"finished_masked_with_chars": f"{i}_deid", 
                                                        "text": f"{i}_original"})
                    
                    elif self.mode=="mask" and self.masking_policy=="fixed_length_chars":
                        pd_deid= deid_res.select("ID", "text", "finished_masked_fixed_length_chars").toPandas()
                        pd_deid= pd_deid.rename(columns={"finished_masked_fixed_length_chars": f"{i}_deid", 
                                                        "text": f"{i}_original"})

                    elif self.mode=="obfuscate":
                        pd_deid= deid_res.select("ID", "text", "finished_obfuscated").toPandas()
                        pd_deid= pd_deid.rename(columns={"finished_obfuscated": f"{i}_deid", 
                                                        "text": f"{i}_original"})

                except:
                    raise Exception("You entered an invalid deidentification mode or masking policy...")
                    
                
                df = pd.merge(df, pd_deid, on='ID', how='outer')
                df.reset_index(drop=True, inplace=True)

                print(f"deidentification process of '{i}' field is completed...")

        except:
            raise Exception("You entered either an invalid field name or wrong separator for input csv file...")

        
        df.to_csv(f"{self.output_file_path}", index=False)
        print(f"Deidentifcation successfully completed and the results saved as '{self.output_file_path}' !")

        df= df.fillna("NONE")
        return_df= self.spark.createDataFrame(df)  

        return return_df


    def deidentify(self, custom_pipeline=None, field_names=["text"], ner_chunk="ner_chunk", sentence="sentence", token="token", document="document", mode="mask", 
                                masking_policy="entity_labels", fixed_mask_length=4, obfuscate_date=True, obfuscate_ref_source="faker", obfuscate_ref_file_path=None, 
                                age_group_obfuscation=False, age_ranges=[1, 4, 12, 20, 40, 60, 80], shift_days=False, number_of_days=None,
                                documentHashCoder_col_name="documentHash", date_tag="DATE", language="en", region="us", 
                                unnormalized_date=False, unnormalized_mode="mask", id_column_name="id", date_shift_column_name="dateshift",
                                separator="\t", input_file_path=None, output_file_path="deidentified.csv"):


        """ This function deidentifies the input file according to the given field names and saves the results as a csv/json file.

        Parameters
        ----------
        custom_pipeline : str
            custom pipeline to be used for deidentification, by default None
        ner_chunk : str, optional
            final chunk column name of custom pipeline that will be deidentified, by default "ner_chunk" 
        field_names : list, optional
            column names that will be deidentified, by default ["text"]
        sentence : str, optional
            sentence column name of the given custom pipeline, by default "sentence"
        token : str, optional
            token column name of the given custom pipeline, by default "token"
        document : str, optional
            document column name of the given custom pipeline, by default "document"
        mode : str, optional
            mode of deidentification, by default "mask"
        masking_policy : str, optional
            masking policy, by default "entity_labels"
        fixed_mask_length : int, optional
            fixed mask length, by default 4
        obfuscate_date : bool, optional
            obfuscate date, by default True
        obfuscate_ref_source : str, optional
            obfuscate reference source, by default "faker"
        obfuscate_ref_file_path : str, optional 
            obfuscate reference file path, by default None
        age_group_obfuscation : bool, optional
            age group obfuscation, by default False
        age_ranges : list, optional  
            age ranges for obfuscation, by default [1, 4, 12, 20, 40, 60, 80]
        shift_days : bool, optional
            shift days, by default False
        number_of_days : int, optional
            number of days, by default None
        documentHashCoder_col_name : str, optional
            document hash coder column name, by default "documentHash" 
        date_tag : str, optional
            date tag, by default "DATE"
        language : str, optional
            language, by default "en"
        region : str, optional
            region, by default "us"
        unnormalized_date : bool, optional
            unnormalized date, by default False
        unnormalized_mode : str, optional
            unnormalized mode, by default "mask"
        id_column_name : str, optional
            ID column name, by default "id"
        date_shift_column_name : str, optional
            date shift column name, by default "date_shift"
        separator : str, optional  
            separator of input csv file, by default "\t"
        input_file_path : str, optional
            input file path, by default None
        output_file_path : str, optional
            output file path, by default 'deidentified.csv' 

        Returns
        -------
        Spark DataFrame
            Spark DataFrame with deidentified text
        csv/json file 
            A deidentified file.

            """


        self.custom_pipeline = custom_pipeline
        self.ner_chunk = ner_chunk
        self.sentence = sentence
        self.token = token
        self.document= document
        self.mode = mode
        self.masking_policy = masking_policy
        self.fixed_mask_length = fixed_mask_length
        self.obfuscate_date = obfuscate_date
        self.obfuscate_ref_source = obfuscate_ref_source
        self.obfuscate_ref_file_path = obfuscate_ref_file_path
        self.age_group_obfuscation = age_group_obfuscation
        self.age_ranges = age_ranges
        self.shift_days = shift_days
        self.number_of_days = number_of_days
        self.documentHashCoder_col_name = documentHashCoder_col_name
        self.date_tag = date_tag
        self.language = language
        self.region = region
        self.unnormalized_date = unnormalized_date
        self.unnormalized_mode = unnormalized_mode
        self.id_column_name = id_column_name
        self.date_shift_column_name = date_shift_column_name


        self.field_names = field_names
        self.separator= separator
        self.input_file_path= input_file_path
        self.custom_pipeline= custom_pipeline
        self.output_file_path= output_file_path 


        # check if the given mode is valid
        mode_list= ["mask", "obfuscate"]
        try:
            if self.mode not in mode_list:
                raise ValueError("You entered an invalid mode option. Please enter 'mask' or 'obfuscate'...")
        except ValueError as e:
            print(e)
            

        # check if the given unnormalized mode is valid
        unnormalized_mode_list= ["mask", "obfuscate"]
        try:
            if self.unnormalized_mode not in unnormalized_mode_list:
                raise ValueError("You entered an invalid unnormalized mode option. Please enter 'mask' or 'obfuscate'...")
        except ValueError as e:
            print(e)


        # check if the given masking policy is valid
        masking_policy_list= ["entity_labels", "same_length_chars", "fixed_length_chars"]
        try:
            if self.masking_policy not in masking_policy_list:
                raise ValueError("You entered an invalid masking policy option. Please enter 'entity_labels', 'same_length_chars' or 'fixed_length_chars'...") 
        except ValueError as e:
            print(e)
 
            
            

        

        if self.custom_pipeline is None:
            deid_pipeline= self.deid_with_pp()

        elif self.custom_pipeline is not None:
            deid_pipeline= self.deid_with_custom_pipeline()

        return deid_pipeline


        #### ------------------------------------- StructuredDeidentification ------------------------------------- ####



    def structured_deidentifier(self, input_file_path=None, output_file_path="deidentified.csv", separator=",",
                                columns_dict={"NAME":"PATIENT","AGE":"AGE"}, ref_source="faker", 
                                obfuscateRefFile=None, columns_seed=None, shift_days=None, 
                                date_formats=["dd/MM/yyyy", "dd-MM-yyyy", "d/M/yyyy", "dd-MM-yyyy", "d-M-yyyy"]):

        
        """ This method is used to deidentify structured data. It takes the input as a file path and returns a deidentified dataframe and a file in csv/json format. 
        
            Parameters: 
            ----------
            
            input_file_path (str): The path of the input file.
            output_file_path (str): The path of the output file.
            separator (str): The seperator of the input csv file.
            columns_dict (dict): A dictionary that contains the column names and the tags that should be used for deidentification.
            ref_source (str): The source of the reference file. It can be "faker" or "file".
            obfuscateRefFile (str): The path of the reference file for obfuscation.
            columns_seed (int): The seed value for the random number generator.
            shift_days (int): The number of days to be shifted.
            date_formats (list): A list of date formats.
            
            Returns:
            -------
            Spark DataFrame: A deidentified dataframe.
            csv/json file: A deidentified file.

            """

        self.input_file_path= input_file_path
        self.output_file_path= output_file_path
        self.separator= separator
        self.columns_dict= columns_dict
        self.ref_source= ref_source
        self.obfuscateRefFile= obfuscateRefFile
        self.columns_seed= columns_seed
        self.shift_days= shift_days
        self.date_formats= date_formats


            
        if self.ref_source=="faker" and self.shift_days==None:
            obfuscator = StructuredDeidentification(self.spark, 
                                                    columns=self.columns_dict, 
                                                    obfuscateRefSource=self.ref_source,
                                                    columnsSeed=self.columns_seed,
                                                    dateFormats= self.date_formats)
            
        elif ref_source=="file":
            obfuscator = StructuredDeidentification(self.spark,
                                                    columns= self.columns_dict, 
                                                    obfuscateRefFile = self.obfuscateRefFile,
                                                    obfuscateRefSource = self.ref_source,
                                                    columnsSeed= self.columns_seed,
                                                    dateFormats= self.date_formats)
            
        elif ref_source=="faker" and shift_days!=None:
            obfuscator = StructuredDeidentification(self.spark, 
                                            columns= self.columns_dict, 
                                            obfuscateRefSource= self.ref_source,
                                            columnsSeed= self.columns_seed,
                                            days= self.shift_days,
                                            dateFormats= self.date_formats)


        try:
            input_file_type= self.input_file_path.split(".")[-1]
            if input_file_type=="csv":
                data = self.spark.read.csv(self.input_file_path, header=True, sep=self.separator)

            elif input_file_type=="json":
                data= self.spark.read.format(input_file_type).load(self.input_file_path)
        
        except:
            raise ValueError("You entered an invalid file path or file format...")
            

        results_df = obfuscator.obfuscateColumns(data)
        results_df_pd= results_df.toPandas()

        if input_file_type=="csv":
            results_df_pd.to_csv(f"{self.output_file_path}", index=False)
            print(f"Deidentifcation successfully completed and the results saved as '{self.output_file_path}' !")


        elif input_file_type=="json":
            results_df_pd.to_json(f"{self.output_file_path}", orient="records")
            print(f"Deidentifcation successfully completed and the results saved as '{self.output_file_path}' !")
        

        return results_df




        


        

        











        
