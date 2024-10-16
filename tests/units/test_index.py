import unittest, sys, os, json
from unittest.mock import patch

from maif_corpus.utils_index import (
    split_format,
    pdf_to_str,
    word_to_str,
    excel_to_table,
    recursive_chunk,
    tiktoken_recursive_chunk,
    spacy_chunk
)


Valid_PDF_MIME = "application/pdf"
Valid_Excel_MIME = "application/vnd.ms-excel"
Valid_Docx_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
Invalid_source = "bubu.bu"
Invalid_File_format = "bu"


# File type utils TESTS


class TestSplitFormat(unittest.TestCase):
    def test_split_format(self):
        self.assertEqual(split_format(Valid_PDF_MIME, True), "pdf")
        self.assertEqual(split_format(Valid_PDF_MIME, False), ("application", "pdf"))
        with self.assertRaises(Exception):
            split_format(Invalid_source, False)
        with self.assertRaises(Exception):
            split_format(Invalid_source, True)


# Files content utils
class TestWordToStr(unittest.TestCase):
    def setUp(self):
        # Load test data
        self.doc = open(os.path.join(os.getcwd(), "tests/fixtures/DocTest.docx"), "rb")
        self.expected_result = json.load(
            open(os.path.join(os.getcwd(), "tests/fixtures/ResultDocTest.json"), "rb"), parse_int=int
        )

    def test_valid_word_file(self):
        filetype = split_format(Valid_Docx_MIME, True)
        self.assertEqual(word_to_str(self.doc, filetype)[0], self.expected_result["0"])

    def test_invalid_file_type(self):
        with self.assertRaises(ValueError):
            word_to_str(self.doc, Invalid_File_format)


class TestPDFToStr(unittest.TestCase):
    def setUp(self):
        self.doc = open(os.path.join(os.getcwd(), "tests/fixtures/PdfTest.pdf"), "rb")

    def test_pdf_to_str(self):
        filetype = split_format(Valid_PDF_MIME, True)
        # Test valid type
        self.assertIsInstance(pdf_to_str(self.doc, filetype), dict)
        # Test invalid type
        with self.assertRaises(ValueError):
            pdf_to_str(self.doc, "docx")

    def test_pdf_to_str_filter_option(self):
        _, filetype = split_format(Valid_PDF_MIME, False)

        # Test valid type
        self.assertIsInstance(pdf_to_str(self.doc, filetype), dict)

        # Test invalid type
        with self.assertRaises(ValueError):
            pdf_to_str(self.doc, Invalid_File_format)


class TestExcelToTable(unittest.TestCase):
    def setUp(self):
        # Load test data
        self.doc = open(os.path.join(os.getcwd(), "tests/fixtures/XlsxText.xlsx"), "rb")
        self.expected_result = json.load(
            open(os.path.join(os.getcwd(), "tests/fixtures/ResultXlsxText.json"), "rb"), parse_int=int
        )

    def test_excel_to_table_valid_type(self):
        # Test with a valid type
        filetype = split_format(Valid_Excel_MIME, True)
        result = excel_to_table(self.doc, filetype)
        self.assertEqual(result.keys(), self.expected_result.keys())

    def test_excel_to_table_invalid_type(self):
        with self.assertRaises(ValueError):
            excel_to_table(self.doc, Invalid_File_format)


class TestRecursiveChunk(unittest.TestCase):
    def setUp(self):
        # Load test data
        self.doc = open(os.path.join(os.getcwd(), "tests/fixtures/ChunkTest.txt"), "r").read()
        self.expected_result = open(os.path.join(os.getcwd(), "tests/fixtures/ResultChunkTest.txt"), "r").readlines()

    def test_recursive_chunk(self):
        recursive_coeff = 0.1
        chunk_size = 200
        result = recursive_chunk(self.doc, recursive_coeff, chunk_size)
        self.assertEqual(len(result), len(self.expected_result))


class TestTiktokenRecursiveChunk(unittest.TestCase):
    def test_tiktoken_recursive_chunk(self):
        # Test with a valid input
        s = "This is a test string."
        recursive_coeff = 0.1
        chunk_size = 200

        with patch(
            "langchain.text_splitter.RecursiveCharacterTextSplitter.from_tiktoken_encoder"
        ) as mock_from_tiktoken_encoder:
            mock_from_tiktoken_encoder.return_value.split_text.return_value = ["This is a test", "string."]

            result = tiktoken_recursive_chunk(s, recursive_coeff, chunk_size)

            self.assertEqual(result, ["This is a test", "string."])

# class TestSpacyChunk(unittest.TestCase):
#     def test_spacy_chunk(self):
#         # Test with a valid input
#         s = "This is a test string."

#         with patch(
#             "langchain.text_splitter.SpacyTextSplitter"
#         ) as mock_from_spacy:
#             mock_from_spacy.return_value.split_text.return_value = ["This is a test", "string."]

#             result = spacy_chunk(s)

#             self.assertEqual(result, ["This is a test", "string."])

if __name__ == "__main__":
    unittest.main()
