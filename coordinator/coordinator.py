import logging
import pickle
import warnings
import grpc
import threading

from src.model_aggregator import ModelAggregator
from src.flag_parser import Parser
import federated_pb2
import federated_pb2_grpc


def iterate_global_model(aggregator, remote_addresses, ports):
    remote_addresses = ["localhost:" + str(port) for port in ports] if remote_addresses == [] else remote_addresses
    print(remote_addresses)
    for epoch in range(parameters['global_epochs']):
        thread_list = []
        for i in range(len(remote_addresses)):
            thread = threading.Thread(target=train_hospital_model, args=(remote_addresses[i], aggregator))
            thread_list.append(thread)
        for thread in thread_list:
            thread.start()
        for thread in thread_list:
            thread.join()
    
        aggregator.aggregate()
        print("Completed epoch %d. Aggregated all model weights." % (epoch))
    
    print('Completed all epochs.')

def train_hospital_model(hospital_address, aggregator):
    channel = grpc.insecure_channel(hospital_address)
    stub = federated_pb2_grpc.HospitalStub(channel)
    print((float(aggregator.global_weights['input.weight'][0][1])))
    exit()
    # stub.Initialize(federated_pb2.InitializeReq())
    hospital_model = stub.ComputeUpdatedModel(federated_pb2.Model(weights=pickle.dumps(aggregator.global_model)))

    aggregator.add_hospital_data(pickle.loads(hospital_model.model.weights), hospital_model.training_samples)
    print("Received a set of weights from address: " + hospital_address)

    channel.close()

if __name__ == "__main__":
    # This prevents the following error message: "pickle support for Storage will be removed in 1.5. Use `torch.save` instead" from being printed to stdout. 
    warnings.filterwarnings("ignore") 
    logging.basicConfig()

    parser = Parser()
    parameters = parser.parse_arguments()
    print('parameters[\'global_epochs\'] = ' + str(parameters['global_epochs']))

    aggregator = ModelAggregator(parameters)

    remote_addresses = parameters['remote_addresses']
    ports = parameters['ports']
    iterate_global_model(aggregator, remote_addresses, ports)
