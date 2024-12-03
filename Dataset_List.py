class DatasetParameters:
    def __init__(self, dataset):
        self.dataset = dataset
        self.parameters = self.get_parameters()

    def get_parameters(self):
        dataset_params = {
            "Stroma": {
                "roi_size": [384,384],
                "input_dim": 3,
                "num_classes": 1,
                "crop" : False,
            },
            "CORN_3_cell": {
                "roi_size": [384,384],
                "input_dim": 3,
                "num_classes": 1,
                "crop" : False
            },
            "CORN_3": {
                "roi_size": [384,384],
                "input_dim": 3,
                "num_classes": 1,
                "crop" : False
            },
            "CORN1": {
                "roi_size": [384,384],
                "dialated_pixels_list": [6,5,4,3,2,1,0.1],
                "input_dim": 3,
                "num_classes": 1,
                "crop" : False
            }

        }
        return dataset_params.get(self.dataset, {})

