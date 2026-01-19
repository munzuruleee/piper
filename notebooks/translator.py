{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e1236f7a-2ef7-4f6b-bc9b-569d77510fa2",
   "metadata": {},
   "outputs": [],
   "source": [
    "pip install piper-tts"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "49df31f8-2230-41c2-9cb3-ba14d3456975",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " config.json\t'epoch=6679-step=1554200.ckpt'\t lightning_logs   m.onnx\n",
      " dataset.jsonl\t'epoch=6681-step=1555028.ckpt'\t m.ckpt\n"
     ]
    }
   ],
   "source": [
    "ls /root/piper/bangla_tts/lightning_logs/version_0/checkpoints"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "54636bdf-bd25-4d8f-89cf-12abdf14961f",
   "metadata": {},
   "outputs": [],
   "source": [
    "cp /root/piper/bangla_tts/config.json /root/piper/bangla_tts/m.onnx.json"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "5461b672-420e-431a-90f3-d2a4d28b536d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "'epoch=6787-step=1598912.ckpt'\n"
     ]
    }
   ],
   "source": [
    " ls /root/piper/bangla_tts/lightning_logs/version_0/checkpoints/\n",
    " cp /root/piper/bangla_tts/lightning_logs/version_0/checkpoints/*.ckpt /root/piper/bangla_tts/m.ckpt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "a5e9c230-3c3b-4c3d-80c3-45b8af56724f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "/root/piper/bangla_tts/m.ckpt\n"
     ]
    }
   ],
   "source": [
    "ls /root/piper/bangla_tts/m.ckpt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "783678fe-da8e-4684-89db-8e9a6da37030",
   "metadata": {},
   "outputs": [],
   "source": [
    "cd /root/piper/src/python"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "7d2666c4-e0de-47bb-8a31-ccd9a4f67071",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "/venv/main/lib/python3.12/site-packages/lightning_fabric/__init__.py:29: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.\n",
      "  __import__(\"pkg_resources\").declare_namespace(__name__)\n",
      "/venv/main/lib/python3.12/site-packages/torch/nn/utils/weight_norm.py:28: UserWarning: torch.nn.utils.weight_norm is deprecated in favor of torch.nn.utils.parametrizations.weight_norm.\n",
      "  warnings.warn(\"torch.nn.utils.weight_norm is deprecated in favor of torch.nn.utils.parametrizations.weight_norm.\")\n",
      "Removing weight norm...\n",
      "/root/piper/src/python/piper_train/vits/attentions.py:235: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!\n",
      "  t_s == t_t\n",
      "/root/piper/src/python/piper_train/vits/attentions.py:295: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!\n",
      "  pad_length = max(length - (self.window_size + 1), 0)\n",
      "/root/piper/src/python/piper_train/vits/attentions.py:296: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!\n",
      "  slice_start_position = max((self.window_size + 1) - length, 0)\n",
      "/root/piper/src/python/piper_train/vits/attentions.py:298: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!\n",
      "  if pad_length > 0:\n",
      "/root/piper/src/python/piper_train/vits/transforms.py:174: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!\n",
      "  assert (discriminant >= 0).all(), discriminant\n",
      "/venv/main/lib/python3.12/site-packages/torch/onnx/_internal/jit_utils.py:307: UserWarning: Constant folding - Only steps=1 can be constant folded for opset >= 10 onnx::Slice op. Constant folding not applied. (Triggered internally at ../torch/csrc/jit/passes/onnx/constant_fold.cpp:179.)\n",
      "  _C._jit_pass_onnx_node_shape_type_inference(node, params_dict, opset_version)\n",
      "/venv/main/lib/python3.12/site-packages/torch/onnx/symbolic_opset10.py:531: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).\n",
      "  return g.op(\"Constant\", value_t=torch.tensor(list_or_value))\n",
      "/venv/main/lib/python3.12/site-packages/torch/onnx/utils.py:702: UserWarning: Constant folding - Only steps=1 can be constant folded for opset >= 10 onnx::Slice op. Constant folding not applied. (Triggered internally at ../torch/csrc/jit/passes/onnx/constant_fold.cpp:179.)\n",
      "  _C._jit_pass_onnx_graph_shape_type_inference(\n",
      "/venv/main/lib/python3.12/site-packages/torch/onnx/utils.py:1208: UserWarning: Constant folding - Only steps=1 can be constant folded for opset >= 10 onnx::Slice op. Constant folding not applied. (Triggered internally at ../torch/csrc/jit/passes/onnx/constant_fold.cpp:179.)\n",
      "  _C._jit_pass_onnx_graph_shape_type_inference(\n",
      "INFO:piper_train.export_onnx:Exported model to /root/piper/bangla_tts/m.onnx\n"
     ]
    }
   ],
   "source": [
    "python3 -m piper_train.export_onnx   /root/piper/bangla_tts/m.ckpt   /root/piper/bangla_tts/m.onnx"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "855c619b-aaec-4a09-8102-672cc5ae26dd",
   "metadata": {},
   "outputs": [],
   "source": [
    "echo \"ইউবোর্ড হলো ইউনিভার্সাল কিবোর্ড\" | piper --model /root/piper/bangla_tts/m.onnx --output_file /root/piper/bangla_tts/test.wav"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "c1c3198e-522e-496e-9759-56fbdb6215c6",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "bash: aplay: command not found\n"
     ]
    },
    {
     "ename": "",
     "evalue": "127",
     "output_type": "error",
     "traceback": []
    }
   ],
   "source": [
    "https://184.191.105.145:21587/edit/root/piper/bangla_tts/test.wav"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "32d0457b-75ae-4a18-aded-0a7ada468d64",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Bash",
   "language": "bash",
   "name": "bash"
  },
  "language_info": {
   "codemirror_mode": "shell",
   "file_extension": ".sh",
   "mimetype": "text/x-sh",
   "name": "bash"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
