from maif_corpus.utils_question import Questions
import pytest
import unittest  
import os  
import pandas as pd 

@pytest.fixture(autouse=True)  
def env_setup(monkeypatch):  
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "mock_api")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_ID", "cog-api-gpt-4o")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "mock_key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://oai-mbe-ao-b2b-sandbox.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_EMBEDDING", "cog-api-embedding")
    monkeypatch.setenv("GRADIO_SERVER_PORT", "7860")
    monkeypatch.setenv("GRADIO_SERVER_NAME", "0.0.0.0")
    monkeypatch.setenv("DOCKER_MODE", "0")

DOCKER_MODE = os.getenv('DOCKER_MODE')

class TestQuestions(unittest.TestCase):  
  
    @classmethod  
    def setUpClass(cls):  
        cls.test_path = "test_ref_question"  
        os.makedirs(cls.test_path, exist_ok=True)  
        cls.questions_data = pd.DataFrame({"Question": 
                                           ["Qui est l'exemple ?", "Est-c'celui qui s'instruit?", "s'détruit?", "en séjournant en taule, en f'sant du mal à autrui ?"]}
                                           )  
      
    def setUp(self):  
        self.q = Questions(self.test_path)  
      
    def tearDown(self):  
        # Cleanup test files  
        for file in os.listdir(self.test_path):  
            os.remove(os.path.join(self.test_path, file))  
      
    @classmethod  
    def tearDownClass(cls):  
        os.rmdir(cls.test_path)  
      
    def test_list_questions_empty(self):  
        self.assertEqual(self.q.list_questions(), [])  
      
    def test_save_questions(self):  
        response = self.q.save_questions("test_questions", self.questions_data)  
        self.assertIn("Votre référentiel test_questions a bien été enregistré", response)  
        self.assertIn("test_questions.yaml", self.q.list_questions())  
      
    def test_load_questions(self):  
        self.q.save_questions("test_questions", self.questions_data)  
        loaded_questions = self.q.load_questions("test_questions.yaml")  
        pd.testing.assert_frame_equal(loaded_questions, self.questions_data)  
      
    def test_delete_questions(self):  
        self.q.save_questions("test_questions", self.questions_data)  
        response = self.q.delete_questions("test_questions.yaml")  
        self.assertIn("Le référentiel test_questions.yaml a bien été supprimé", response)  
        self.assertNotIn("test_questions.yaml", self.q.list_questions())  
  
if __name__ == '__main__':  
    unittest.main()  
