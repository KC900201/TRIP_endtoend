# Traffic Risk Information Provision Network (TRIP)

### Author
Ng Kwong Cheong

### Contributors
1. Masayasu Atsumi (Professor)
2. Yuki Murata

### Description
The Traffic Risk Information Provision Network (TRIP) is a project that estimates traffic risk during 
road navigation based on the spatio-temporal deep neural network (DNN) trained by our novel comparative 
loss function. The purpose of this project is to initate a research method that performs traffic risk
estimation from on-vehicle camera image sequence based on detecting moving objects and extracting the 
moving object regions using an object detection network. Risk estimation experiments were conducted on
a combination of real image datasets and virtually simulated image datasets.

As this project is still on progress, we are going to improve the accuracy of the preliminary risk 
prediction of the model by extending the risk estimation network. In addition, we hope to further
evaluate the effectiveness of the proposed network in simulated environment applications by implementing
the proposed model in a driving simulator.

## Languages used
Python

## Characteristics
1. Region-based convolutional neural network YOLOv2 (latest branch YOLOv3) for moving object detection
2. Spatial pyramid and LSTM-based network for spatio-temporal pattern encoding
3. Comparative loss function based on traffic situation pair comparison for estimation of risk level

## Publications
1. [Performance Enhancement of Region-based Spatio-temporal Neural Network for Traffic Risk Estimation using Real and Virtual Datasets](https://www.jstage.jst.go.jp/article/pjsai/JSAI2020/0/JSAI2020_3F1ES201/_pdf)
2. [Traffic Risk Estimation from On-vehicle Video by Region-based Spatio-temporal DNN trained using Comparative Loss](https://www.jstage.jst.go.jp/article/pjsai/JSAI2019/0/JSAI2019_3Rin201/_pdf)

## Contacts
Ng Kwong Cheong (nkc900201@gmail.com);
Yuki Murata (e19d5202@soka-u.jp);
Masayasu Atsumi (matsumi@soka.ac.jp);

## Contributions
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change for improvement.

Please make sure to update tests as appropriate

## License
[MIT](https://choosealicense.com/licenses/mit/)

