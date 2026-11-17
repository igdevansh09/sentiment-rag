import pandas as pd
from datasets import load_dataset
from hugging_face_authentication import hugging_face_auth


class DatasetLoader:
    DEFAULT_DATASET_ID = "mteb/tweet_sentiment_extraction"
    
    def __init__(self, dataset_id: str = DEFAULT_DATASET_ID):
        hugging_face_auth()
        self.dataset_id = dataset_id
        self.dataset = load_dataset(dataset_id, split="train")
        self.current_column = None
        
    def checking_dataset_as_pandas(self) -> pd.DataFrame:
        dataset = self.dataset.to_pandas()
        if isinstance(dataset, pd.DataFrame):
            return dataset
        return pd.concat(dataset, ignore_index=True)
    
    def printing_dataset(self, min=0, max=None):
        print(self.checking_dataset_as_pandas().iloc[min:max])
        
    def get_more_information(self):
        column_names = self.dataset.column_names
        information = f"""
            this dataset has the {len(self.dataset)} columns and are the following {column_names}
        """
        print(information.strip())
        return information
    
    def ask_user_to_choose_a_column_to_work_with(self):
        user_input = input(f"these are our columns: {self.dataset.column_names} please choose one to work with \n")
        
        if user_input in self.dataset:
            return user_input
        else:
            print("you didn't enter a valid column name we will be adding text instead")
            return 'text'
        
    def ask_user_to_choose_a_column_to_work_on_the_dataset(self):
        user_choice = input(f"these are our columns: {self.dataset.column_names} please choose one to work with \n")

        if user_choice in self.dataset.column_names:
            return user_choice
        else:
            # we will send by default the text in the dataset
            print("you didn't enter a valid column name we will be adding text instead")
            return 'text'
        
    
    def __repr__(self) -> str:
        return (
            f"DatasetLoader("
            f"dataset_id='{self.dataset_id}', "
            f"rows={len(self.dataset)}, "
            f"columns={self.dataset.column_names})"
        )
        
x = DatasetLoader()
x.get_more_information()
