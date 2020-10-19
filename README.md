

<br />
<p align="center">

  <h3 align="center">Fully_connected_multylayer_DNN_to_classify_MNIST10</h3>


  </p>
</p>



<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
  * [Built With](#built-with)
* [Prerequisites](#prerequisites)
* [Contributing](#contributing)
* [License](#license)


<!-- ABOUT THE PROJECT -->
## About The Project
This project is in part, an assignment given in the Deep Learning course "CSCI 315: Artificial Intelligence through Deep Learning" which was offered at BGU by prof. Levy.
The main purpose of the Code presented here is to classify some 500 test examples of the MNIST 10 dataset, by training on 500 examples from a training dataset.

As you'll be able to see in the assignment description pdf, only one layer was required to complete the assignment, however the result is easily generalized in the "MultyLayerBackprop.py" file to include an arbitrary number of hidden layers, chosen by the user. 

Some further educational content may be found in the code presented:
1) The use of "Mumentum" to accelerate the network's convergence requires another tunable hyper parameter. The user will be able to tweek it and observe the changes is convergence rates.
2) In the files included you'll be able to find simulations of the network's training using "online learning" and "Batch learning". The user will be able to get a sense of the convergence rate in each of these methods (which basically determines either if the weights are to be updated on each example individually, or all at the same time at he end of an epoch)

You might notice some "pickle files" i.e. files with postfix ".p". Those are just the data which was parssed from the .txt files of training and test images. These files are unnecessary as there is a "read_data_set_into_array.py" script which will do this job for you. That said, the parssing of these files into individual calagoged numbers is computationaly very ime consuming, and so we provide the parssed data in the form of the pickle files so that in each new run of the program the user wont have to re-parse the data.

### Built With

* [numpy](https://numpy.org/)
* [Python 3.7.1](https://www.python.org/downloads/release/python-371/)
* [Pickle](https://docs.python.org/3/library/pickle.html)




## Prerequisites

Just install the dependencies written above.


<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/nadavleh/repo.svg?style=flat-square
[forks-shield]: https://img.shields.io/github/forks/nadavleh/repo.svg?style=flat-square
[forks-url]: https://github.com/nadavleh/repo/network/members
[stars-shield]: https://img.shields.io/github/stars/nadavleh/repo.svg?style=flat-square
[stars-url]: https://github.com/nadavleh/repo/stargazers
[issues-shield]: https://img.shields.io/github/issues/nadavleh/repo.svg?style=flat-square
[issues-url]: https://github.com/nadavleh/repo/issues
[license-shield]: https://img.shields.io/github/license/nadavleh/repo.svg?style=flat-square
