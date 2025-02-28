"use client";

import React, { useState } from "react";
import axios from "axios";

interface SimilarImage {
  url: string;
  name: string;
}

const VisionSearch = () => {
  const [queryImage, setQueryImage] = useState<File | null>(null);
  const [similarImages, setSimilarImages] = useState<SimilarImage[]>([]);
  const BACKEND_URL = process.env.REACT_APP_BACKEND_URL ;

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setQueryImage(e.target.files[0]);
    }
  };

  const handleSubmit = async () => {
    if (!queryImage) {
      console.error("No image selected");
      return;
    }

    const formData = new FormData();
    formData.append("image", queryImage);

    try {
      const response = await axios.post(`${BACKEND_URL}/vision-search`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      setSimilarImages(response.data.similarImages || []);
    } catch (error) {
      console.error("Error during vision search:", error);
    }
  };

  return (
    <div>
      <h1>Vision Search</h1>
      <input type="file" accept="image/*" onChange={handleImageUpload} />
      <button onClick={handleSubmit}>Search</button>

      <ul>
        {similarImages.map((img, index) => (
          <li key={index}>
            <img src={`${BACKEND_URL}${img.url}`} alt={img.name} width="200" />
            <p>{img.name}</p>
          </li>
        ))}
      </ul>
    </div>
  );
};

export default VisionSearch;
