Explain files/folders :

- models -> put the .pth of your VGT model
- DocLaynet_core -> put here your DocLayNet dataset (with the .pkl files corresponding to the dataset)
- pdf_documents_test -> contains the pdf file for test
- img_documents_output -> contains the .png files converted from the pdf file
- pkl_documents_output -> contains the .pkl files converted from the pdf file
- layoutlm-base-uncased -> model weights for layoutlm (see https://huggingface.co/microsoft/layoutlm-base-uncased)
- useful_cmd -> contains the cmd ligns for converting the pdf/pkl and run an inference
- VGT -> contains the code source for VGT model (see https://github.com/AlibabaResearch/AdvancedLiterateMachinery)
