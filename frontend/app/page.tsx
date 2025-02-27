"use client";

import React, { useState } from 'react';
import axios from 'axios';

const VisionSearch = () => {
    const [queryImage, setQueryImage] = useState<File | null>(null);
    interface SimilarImage {
        url: string;
        name: string;
    }
    
    const [similarImages, setSimilarImages] = useState<SimilarImage[]>([]);

    const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            setQueryImage(e.target.files[0]);
        }
    };

    const handleSubmit = async () => {
        if (!queryImage) return;

        // Send the query image to the backend
        const formData = new FormData();
        formData.append('image', queryImage);

        try {
            const response = await axios.post('https://33c5-35-240-223-21.ngrok-free.app/vision-search',
                formData,
                {
                    headers: { 'Content-Type': 'multipart/form-data' },
                }
            );

            // Update the state with similar images
            setSimilarImages(response.data.similarImages);
        } catch (error) {
            console.error('Error during vision search:', error);
        }
    };

    return (
        <div>
            <h1>Vision Search</h1>
            <input type="file" onChange={handleImageUpload} />
            <button onClick={handleSubmit}>Search</button>

            {similarImages.map((img, index) => (
                <li key={index}>
                    {/* Prepend the backend URL to the image path */}
                    <img
                        src={`https://33c5-35-240-223-21.ngrok-free.app${img.url}`}
                        alt={`Similar ${index + 1}`}
                        width="200"
                    />
                    <p>{img.name}</p>
                </li>
            ))}
        </div>
    );
};

export default VisionSearch;