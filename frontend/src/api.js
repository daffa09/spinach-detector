import axios from "axios";

export const detectImage = async (image, model) => {
  const formData = new FormData();
  formData.append("image", image);
  formData.append("model_name", model);

  const res = await axios.post("http://127.0.0.1:5000/predict", formData);
  return res.data;
};
