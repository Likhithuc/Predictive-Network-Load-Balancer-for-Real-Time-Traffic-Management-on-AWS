import sagemaker
from sagemaker import get_execution_role
from sagemaker.tensorflow import TensorFlowModel

role = get_execution_role()
model_path = "s3://your-sagemaker-model-bucket/smart_scaling_model.h5"

tensorflow_model = TensorFlowModel(model_data=model_path,
                                   role=role,
                                   framework_version='2.11',
                                   entry_point='inference.py')  # You’ll define this file

predictor = tensorflow_model.deploy(
    instance_type='ml.m5.large',
    initial_instance_count=1
)
