# kervolution
Kervolution Library in PyTorch

# Usage
It is suggested to add this repo as a submodule in your project.

    git submodule add https://github.com/wang-chen/kervolution [target folder]
    
Then you can replace nn.Conv by nn.Kerv directly in you python script.

You may also need the following commands.

    git submodule init
    git submodule update

# Citation

    @inproceedings{wang2019kervolutional,
      title={Kervolutional Neural Networks},
      author={Wang, Chen and Yang, Jianfei and Xie, Lihua and Yuan, Junsong},
      booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
      pages={31--40},
      year={2019}
    }
