import React, { useState } from 'react';
import axios from 'axios';

const VisionSearch = () => {
    const [queryImage, setQueryImage] = useState(null);
    const [similarImages, setSimilarImages] = useState([]);

    const handleImageUpload = (e) => {
        setQueryImage(e.target.files[0]);
    };

    const handleSubmit = async () => {
        if (!queryImage) return;

        // Send the query image to the backend
        const formData = new FormData();
        formData.append('image', queryImage);

        try {
            const response = await axios.post(
                'https://your-backend-url/vision-search',
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

            {similarImages.length > 0 && (
                <div>
                    <h2>Top Similar Images:</h2>
                    <ul>
                        {similarImages.map((img, index) => (
                            <li key={index}>
                                <img src={img.url} alt={`Similar ${index + 1}`} width="200" />
                                <p>{img.name}</p>
                            </li>
                        ))}
                    </ul>
                </div>
            )}
        </div>
    );
};

export default VisionSearch;