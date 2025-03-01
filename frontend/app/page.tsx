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
  const [useRoboflow, setUseRoboflow] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const BACKEND_URL = "http://127.0.0.1:5000";

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setQueryImage(e.target.files[0]);
    }
  };

  const handleSubmit = async () => {
    if (!queryImage) {
      setError("Please select an image.");
      return;
    }

    setLoading(true);
    setError(null);
    setSimilarImages([]);

    const formData = new FormData();
    formData.append("image", queryImage);

    try {
      const endpoint = useRoboflow ? "/vision-search-roboflow" : "/vision-search-hf";
      const response = await axios.post(`${BACKEND_URL}${endpoint}`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      setSimilarImages(response.data.similarImages || []);
    } catch (err) {
      setError("An error occurred while processing the image.");
      console.error("Error during vision search:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.container}>
      <h1 style={styles.heading}>Vision Search</h1>
      <div style={styles.mainLayout}>
        {/* Left Panel: Upload and Options */}
        <div style={styles.leftPanel}>
          <h2 style={styles.subHeading}>Upload an Image</h2>
          <input
            type="file"
            accept="image/*"
            onChange={handleImageUpload}
            style={styles.fileInput}
          />
          <div style={styles.radioGroup}>
            <label style={styles.radioLabel}>
              <input
                type="radio"
                name="model"
                value="hf"
                checked={!useRoboflow}
                onChange={() => setUseRoboflow(false)}
              />
              Hugging Face
            </label>
            <label style={styles.radioLabel}>
              <input
                type="radio"
                name="model"
                value="roboflow"
                checked={useRoboflow}
                onChange={() => setUseRoboflow(true)}
              />
              Roboflow
            </label>
          </div>
          <button onClick={handleSubmit} style={styles.searchButton} disabled={loading}>
            {loading ? "Searching..." : "Search"}
          </button>
          {error && <p style={styles.errorMessage}>{error}</p>}
        </div>

        {/* Right Panel: Similar Images */}
        <div style={styles.rightPanel}>
          <h2 style={styles.subHeading}>Similar Images</h2>
          {loading ? (
            <p style={styles.loadingMessage}>Loading...</p>
          ) : similarImages.length > 0 ? (
            <ul style={styles.imageList}>
              {similarImages.map((img, index) => (
                <li key={index} style={styles.imageItem}>
                  <img src={`${BACKEND_URL}${img.url}`} alt={img.name} style={styles.image} />
                  <p style={styles.imageName}>{img.name}</p>
                </li>
              ))}
            </ul>
          ) : (
            <p style={styles.noResults}>No similar images found.</p>
          )}
        </div>
      </div>
    </div>
  );
};

// Styles
import { CSSProperties } from "react";

const styles: { [key: string]: CSSProperties } = {
  container: {
    fontFamily: "Arial, sans-serif",
    padding: "20px",
    maxWidth: "1200px",
    margin: "0 auto",
  },
  heading: {
    textAlign: "center",
    fontSize: "2rem",
    marginBottom: "20px",
  },
  mainLayout: {
    display: "flex",
    gap: "40px",
    alignItems: "flex-start",
  },
  leftPanel: {
    width: "45%",
    border: "1px solid #ccc",
    borderRadius: "8px",
    padding: "20px",
    boxShadow: "0 2px 4px rgba(0, 0, 0, 0.1)",
  },
  subHeading: {
    fontSize: "1.5rem",
    marginBottom: "15px",
  },
  fileInput: {
    marginBottom: "15px",
    padding: "8px",
    border: "1px solid #ccc",
    borderRadius: "4px",
    width: "100%",
  },
  radioGroup: {
    marginBottom: "15px",
  },
  radioLabel: {
    marginRight: "15px",
    fontSize: "1rem",
  },
  searchButton: {
    padding: "10px 20px",
    backgroundColor: "#007BFF",
    color: "#fff",
    border: "none",
    borderRadius: "4px",
    cursor: "pointer",
    fontSize: "1rem",
  },
  errorMessage: {
    color: "red",
    marginTop: "10px",
    fontSize: "1rem",
  },
  rightPanel: {
    width: "45%",
    border: "1px solid #ccc",
    borderRadius: "8px",
    padding: "20px",
    boxShadow: "0 2px 4px rgba(0, 0, 0, 0.1)",
  },
  imageList: {
    listStyle: "none",
    padding: 0,
    display: "grid",
    gridTemplateColumns: "repeat(auto-fill, minmax(150px, 1fr))",
    gap: "15px",
  },
  imageItem: {
    textAlign: "center",
  },
  image: {
    width: "100%",
    height: "150px",
    objectFit: "cover",
    borderRadius: "4px",
    marginBottom: "5px",
  },
  imageName: {
    fontSize: "0.9rem",
    color: "#555",
  },
  loadingMessage: {
    textAlign: "center",
    fontSize: "1rem",
    color: "#555",
  },
  noResults: {
    textAlign: "center",
    fontSize: "1rem",
    color: "#555",
  },
};

export default VisionSearch;
