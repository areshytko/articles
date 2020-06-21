# Distributed deep learning experiments with Ansible, AWS and Pytorch Lightning. Part 1

source code materials for the blog article

### To run:

1. Add default VPC ID to `config.yaml`
2. Add registered SSH key pair to `config.yaml`
3. Setup dependencies: `pip install -r requirements.txt`
4. Create a cluster: `ansible-playbook setup-play.yml`
5. Submit a training procedure: `./submit.py -- ddp_train_example.py gpus=1 num_nodes=2 distributed_backend=ddp`
6. Terminate the cluster: `ansible-playbook -i aws_ec2.yaml cleanup-play.yml`
