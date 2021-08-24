import onnx
import onnxruntime
import onnx.helper as helper
import numpy as np
import onnxruntime.tools.symbolic_shape_infer as sym

from onnx import TensorProto, shape_inference, numpy_helper

position = '3329'
data =np.zeros(1).squeeze().astype(np.int64)
posIntializer = numpy_helper.from_array(data, name=position)

idx = '3335'
# data =np.zeros(1).squeeze().astype(np.int64)
idxIntializer = numpy_helper.from_array(data, name=idx)

idx2 = '3357'
data_idx2 =np.ones(1).squeeze().astype(np.int64)
idx2Intializer = numpy_helper.from_array(data_idx2, name=idx2)

inputs = '3967'
inputs_data =np.zeros(1).astype(np.int64)
inputsIntializer = numpy_helper.from_array(inputs_data, name=inputs)

axes = 'axes'
axes_data =np.zeros(1).astype(np.int64)
axIntializer = numpy_helper.from_array(axes_data, name=axes)

cond = '3328'
cond_data =np.array([True], dtype=np.bool).squeeze()
condIntializer = numpy_helper.from_array(cond_data, name=cond)

body = helper.make_graph(
    [
        helper.make_node("SequenceAt", ["3354", "i_1"], ["3367"]),
        helper.make_node("ReduceMax", ["3367"], ["3368"]),
        helper.make_node("Concat", ["pooling_tensors.23", "3368"], ["3369"], axis =0),
        helper.make_node("Identity", ["3328"], ["3380"])
    ],
    "body",
    [
        helper.make_tensor_value_info('i_1', TensorProto.INT64, [1]),
        helper.make_tensor_value_info('cond', TensorProto.BOOL, []),
        helper.make_tensor_value_info('pooling_tensors.23', TensorProto.FLOAT, ['p0', 'p1'])
    ],
    [
        helper.make_tensor_value_info('3380', TensorProto.BOOL, []),
        helper.make_tensor_value_info('3369', TensorProto.FLOAT, ['3369_d0', '3369_d1'])
    ]
)

# Create the outer network
graph_proto = helper.make_graph(
    [
        helper.make_node("SplitToSequence", ["3351", "3353"], ["3354"], axis =0, keepdims=1),
        helper.make_node("SequenceAt", ["3354", "3329"], ["3355"]),
        helper.make_node("Shape", ["3355"], ["3356"]),
        helper.make_node("Gather", ["3356", "3357"], ["3358"], axis=0),
        helper.make_node("Unsqueeze", ["3358", "axes"], ["3360"]),
        helper.make_node("Concat", ["3967", "3360"], ["3361"], axis=0),
        helper.make_node("ConstantOfShape", ["3361"], ["3362"]),
        helper.make_node("Gather", ["3334", "3335"], ["3336"], axis=0),
        helper.make_node("Loop", ["3336", "3328", "3362"],
                         ["3363"], body=body)
    ],
    "outer",
    [
        helper.make_tensor_value_info('3351', TensorProto.FLOAT, ['3351_d1',1]),
        helper.make_tensor_value_info('3353', TensorProto.INT32, ['3353_d0']),
        helper.make_tensor_value_info('3334', TensorProto.INT64, [1]),
    ],
    [
        helper.make_tensor_value_info('3363', TensorProto.FLOAT, ['3363_d0', '3363_d1'])
    ],
    [posIntializer, idxIntializer, idx2Intializer, inputsIntializer, condIntializer, axIntializer]
)

model = helper.make_model(graph_proto)
onnx.checker.check_model(model)
print("model checked")


optimization_passes = [
    "eliminate_deadend",
    "eliminate_identity",
    "eliminate_nop_dropout",
    "eliminate_nop_monotone_argmax",
    "eliminate_nop_pad",
    "eliminate_nop_transpose",
    "eliminate_unused_initializer",
    "fuse_add_bias_into_conv",
    "fuse_bn_into_conv",
    "fuse_consecutive_concats",
    "fuse_consecutive_log_softmax",
    "fuse_consecutive_reduce_unsqueeze",
    "fuse_consecutive_squeezes",
    "fuse_matmul_add_bias_into_gemm",
    "fuse_pad_into_conv",
    "fuse_transpose_into_gemm",
    "fuse_consecutive_transposes"]

# do const folding
so = onnxruntime.SessionOptions()
so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
so.optimized_model_filepath = "seq_loop_cf.onnx"
sess = onnxruntime.InferenceSession(model.SerializeToString(), sess_options=so)

data_3351 = np.random.rand(3, 1).astype(np.float32)
data_3353 = np.array([1, 1, 1]).astype(np.int32)
data_3334 = np.ones((1)).astype(np.int64)

result = sess.run(None, {"3351": data_3351, "3353": data_3353, "3334": data_3334})
print("result", result)
