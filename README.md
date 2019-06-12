# Real-Time-Iris-Tracking
<p>A program to track iris position and motion from a live webcam stream.</p>

<h2>How it works:</h2>
<p>Using dlib's landmark extractor, we can extract the eye region from a detected face. Sliding a filter through the eye sclera, the iris region can be detected, and the pupil and iris radius are automatically inferred. For visual representation, we mark the detected iris with a circle.
  
<h2>Frameworks/Libraries Used</h2>
	<ul>
		<li>Python</li>
		<li>OpenCV</li>
		<li>DLIB</li>
		<li>Anaconda distribution</li>
	</ul>
