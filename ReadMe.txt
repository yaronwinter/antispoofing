Anti Spoofing System

Installation Instruction:

(1) Clone the repository https://github.com/yaronwinter/antispoofing.
(2) Use the docker file (beneath the root folder) for generating the image.
(3) Run the container induced by the generated image.
	- Make sure to mount a host folder to the volumed container folder (/app/data)
(4) Copy the three zipped datasets to the mounted folder and unzip them.
	- The zipped datasets  are shared from a google drive:
		* sub_sample.zip  - a subset that contain ~80%, randomly, of the LA dataset
		* toy_sample.zip  - a subset that contain ~10%, randomly, of the LA dataset
		* cold_sample.zip - a few dozens example, for correctness verification test
(5) When all installed, the system support these four functions:
	(i)   train cnn model:    #python app.py -config configs/cnn.json -func_name train_cnn 
	(ii)  train aasist model: #python app.py -config configs/aasist.json -func_name train_aasist 
	(iii) evaluate sample:    #python app.py -config configs/<model name>.json -func_name evaluate_sample -model_path <model path> -model_type <model name> -protocol <protocol file> -audio_folder <audio folder>
	(iv)  evaluate signal:    #python app.py -config configs/<model name>.json -func_name evaluate_signal -model_path <model path> -model_type <model name> -label <label> -attack <attack type>
	- Where:
		* model name   := cnn / aasist
		* model path   := a full path to a model. The models are generated during training
		* protocol     := a file formatted as AVS 2019 protocol file
		* audio folder := a folder containing audio file. Must be associated with the protocol file (as in AVS 2019 dataset)
		* label        := bonafide / spoof / None
		* attack       := one of the AVS 2019 attack codes, or None
	- Functions Outputs:
		* train (cnn / aasist): models iamges are stored in /app/model during the training
		* evaluate sample:      output log files is written to /app/logs, describing the model's results for each evaluated item
		* evaluate signal:      output the scores to the terminal screen
		
* Important Comments:
  (1) The code support both cpu and cuda
  (2) As my machine doesn't have GPU - it was not actually tested on cuda.
  (3) The notebook version did run on GPU, through Google Colab.
  (4) The smaller sub samples were used due to limited resources constraints.

