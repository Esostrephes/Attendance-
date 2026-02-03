from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import numpy as np
from insightface.app import FaceAnalysis
import base64
import uvicorn
from threading import Lock
import os
import firebase_admin
from firebase_admin import credentials, firestore
import uuid
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- FIREBASE INITIALIZATION --------------------
try:
    firebase_creds = os.environ.get('FIREBASE_CREDENTIALS')
    if firebase_creds:
        cred_dict = json.loads(firebase_creds)
        cred = credentials.Certificate(cred_dict)
    else:
        cred = credentials.Certificate("esosystem-48ef8-firebase-adminsdk-fbsvc-fa8acf9a93.json")
    
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("✅ Firebase initialized successfully!")
except Exception as e:
    print(f"❌ Firebase initialization error: {e}")
    db = None

# -------------------- GLOBAL STATE --------------------
class EnrollmentState:
    def __init__(self):
        self.face_app = None
        self.POSE_BINS = 8
        self.SIMILARITY_THRESHOLD = 0.6
        self.ALPHA = 0.3
        self.pose_filled = [False] * self.POSE_BINS
        self.bin_attempts = [0] * self.POSE_BINS
        self.identity_embedding = None
        self.count = 0
        self.is_enrolling = False
        self.last_message = ""
        self.current_bin = -1
        self.current_user_id = None
        self.current_user_name = None
        self.current_user_email = None
        self.current_metadata = None
        self.lock = Lock()
        
    def reset(self, user_id=None, user_name=None, user_email=None, metadata=None):
        with self.lock:
            self.pose_filled = [False] * self.POSE_BINS
            self.bin_attempts = [0] * self.POSE_BINS
            self.identity_embedding = None
            self.count = 0
            self.is_enrolling = True
            self.last_message = "Starting enrollment..."
            self.current_bin = -1
            self.current_user_id = user_id or str(uuid.uuid4())
            self.current_user_name = user_name
            self.current_user_email = user_email
            self.current_metadata = metadata or {}

state = EnrollmentState()

# -------------------- PYDANTIC MODELS --------------------
class FrameData(BaseModel):
    image: str

class EnrollmentRequest(BaseModel):
    user_id: str = None
    user_name: str = None
    user_email: str = None
    metadata: dict = None

# -------------------- FIREBASE HELPERS --------------------
def save_embedding_to_firestore(embedding, user_id, user_name=None, user_email=None, metadata=None):
    try:
        if db is None:
            raise Exception("Firestore not initialized")
        
        embedding_list = embedding.tolist()
        
        doc_data = {
            'user_id': user_id,
            'embedding': embedding_list,
            'embedding_shape': list(embedding.shape),
            'created_at': firestore.SERVER_TIMESTAMP,
            'updated_at': firestore.SERVER_TIMESTAMP,
            'samples_count': state.count,
            'pose_bins': state.POSE_BINS,
        }
        
        if user_name:
            doc_data['user_name'] = user_name
        if user_email:
            doc_data['user_email'] = user_email
        if metadata:
            doc_data['metadata'] = metadata
        
        doc_ref = db.collection('face_embeddings').document(user_id)
        doc_ref.set(doc_data)
        
        print(f"✅ Embedding saved to Firestore for user: {user_id}")
        return True
    except Exception as e:
        print(f"❌ Error saving to Firestore: {e}")
        raise e

def update_embedding_in_firestore(embedding, user_id):
    try:
        if db is None:
            raise Exception("Firestore not initialized")
        
        embedding_list = embedding.tolist()
        
        doc_ref = db.collection('face_embeddings').document(user_id)
        doc_ref.update({
            'embedding': embedding_list,
            'updated_at': firestore.SERVER_TIMESTAMP,
            'samples_count': state.count,
        })
        
        print(f"✅ Embedding updated in Firestore for user: {user_id}")
        return True
    except Exception as e:
        print(f"❌ Error updating Firestore: {e}")
        raise e

def get_embedding_from_firestore(user_id):
    try:
        if db is None:
            raise Exception("Firestore not initialized")
        
        doc_ref = db.collection('face_embeddings').document(user_id)
        doc = doc_ref.get()
        
        if doc.exists:
            data = doc.to_dict()
            embedding = np.array(data['embedding'])
            return {
                'embedding': embedding,
                'user_id': data['user_id'],
                'user_name': data.get('user_name'),
                'user_email': data.get('user_email'),
                'created_at': data.get('created_at'),
                'updated_at': data.get('updated_at'),
                'metadata': data.get('metadata'),
                'samples_count': data.get('samples_count')
            }
        else:
            return None
    except Exception as e:
        print(f"❌ Error retrieving from Firestore: {e}")
        raise e

def get_all_embeddings():
    try:
        if db is None:
            raise Exception("Firestore not initialized")
        
        embeddings_ref = db.collection('face_embeddings')
        docs = embeddings_ref.stream()
        
        all_embeddings = []
        for doc in docs:
            data = doc.to_dict()
            all_embeddings.append({
                'user_id': data['user_id'],
                'user_name': data.get('user_name'),
                'user_email': data.get('user_email'),
                'created_at': data.get('created_at'),
                'samples_count': data.get('samples_count')
            })
        
        return all_embeddings
    except Exception as e:
        print(f"❌ Error getting all embeddings: {e}")
        raise e

def search_similar_embeddings(query_embedding, threshold=0.6, limit=10):
    try:
        if db is None:
            raise Exception("Firestore not initialized")
        
        embeddings_ref = db.collection('face_embeddings')
        docs = embeddings_ref.stream()
        
        matches = []
        query_emb = np.array(query_embedding) if isinstance(query_embedding, list) else query_embedding
        
        for doc in docs:
            data = doc.to_dict()
            stored_emb = np.array(data['embedding'])
            
            similarity = 1 - cosine_distance(query_emb, stored_emb)
            
            if similarity >= threshold:
                matches.append({
                    'user_id': data['user_id'],
                    'user_name': data.get('user_name'),
                    'user_email': data.get('user_email'),
                    'similarity': float(similarity),
                    'created_at': data.get('created_at')
                })
        
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        return matches[:limit]
    except Exception as e:
        print(f"❌ Error searching embeddings: {e}")
        raise e

# -------------------- HELPERS --------------------
def l2_normalize(x):
    return x / np.linalg.norm(x)

def cosine_distance(a, b):
    return 1 - np.dot(a, b)

def init_face_model():
    if state.face_app is None:
        print("Loading InsightFace model...")
        state.face_app = FaceAnalysis(name="buffalo_l")
        state.face_app.prepare(ctx_id=-1, det_size=(640, 640))
        print("✅ Model loaded successfully!")

# -------------------- PROCESSING --------------------
def process_enrollment_frame(frame):
    with state.lock:
        if state.face_app is None:
            return {'status': 'error', 'message': 'Model not initialized'}
        
        faces = state.face_app.get(frame)
        
        if len(faces) == 0:
            state.last_message = "No face detected - move closer"
            return {
                'status': 'no_face',
                'message': state.last_message,
                'progress': (sum(state.pose_filled) / state.POSE_BINS) * 100,
                'pose_filled': state.pose_filled,
                'current_bin': -1
            }
        
        if len(faces) > 1:
            state.last_message = "Multiple faces - show only one person"
            return {
                'status': 'multiple_faces',
                'message': state.last_message,
                'progress': (sum(state.pose_filled) / state.POSE_BINS) * 100,
                'pose_filled': state.pose_filled,
                'current_bin': -1
            }
        
        face = faces[0]
        
        if not hasattr(face, 'pose') or face.pose is None:
            state.last_message = "Pose estimation unavailable"
            return {
                'status': 'no_pose',
                'message': state.last_message,
                'progress': (sum(state.pose_filled) / state.POSE_BINS) * 100,
                'pose_filled': state.pose_filled,
                'current_bin': -1
            }
        
        yaw = face.pose[1]
        bin_index = int(((yaw + 90) / 180) * state.POSE_BINS)
        bin_index = max(0, min(state.POSE_BINS - 1, bin_index))
        state.current_bin = bin_index
        
        emb = l2_normalize(face.embedding)
        
        if state.identity_embedding is not None:
            similarity = 1 - cosine_distance(emb, state.identity_embedding)
            if similarity < state.SIMILARITY_THRESHOLD:
                state.last_message = "Different person detected!"
                return {
                    'status': 'different_person',
                    'message': state.last_message,
                    'progress': (sum(state.pose_filled) / state.POSE_BINS) * 100,
                    'pose_filled': state.pose_filled,
                    'current_bin': bin_index
                }
        
        if not state.pose_filled[bin_index]:
            state.pose_filled[bin_index] = True
            
            if state.identity_embedding is None:
                state.identity_embedding = emb
                state.count = 1
                state.last_message = f"Bin {bin_index + 1}/{state.POSE_BINS} captured"
                
                try:
                    save_embedding_to_firestore(
                        state.identity_embedding,
                        state.current_user_id,
                        state.current_user_name,
                        state.current_user_email,
                        state.current_metadata
                    )
                except Exception as e:
                    print(f"Error saving initial embedding: {e}")
            else:
                state.identity_embedding = state.ALPHA * emb + (1 - state.ALPHA) * state.identity_embedding
                state.identity_embedding = l2_normalize(state.identity_embedding)
                state.count += 1
                state.last_message = f"Bin {bin_index + 1}/{state.POSE_BINS} captured"
                
                try:
                    update_embedding_in_firestore(
                        state.identity_embedding,
                        state.current_user_id
                    )
                except Exception as e:
                    print(f"Error updating embedding: {e}")
        
        state.bin_attempts[bin_index] += 1
        if state.bin_attempts[bin_index] > 150 and not state.pose_filled[bin_index]:
            state.bin_attempts[bin_index] = 0
        
        filled_count = sum(state.pose_filled)
        is_complete = filled_count == state.POSE_BINS
        
        if is_complete:
            state.last_message = "🎉 Enrollment complete!"
            state.is_enrolling = False
            
            try:
                update_embedding_in_firestore(
                    state.identity_embedding,
                    state.current_user_id
                )
            except Exception as e:
                print(f"Error in final update: {e}")
        
        return {
            'status': 'success',
            'message': state.last_message,
            'progress': (filled_count / state.POSE_BINS) * 100,
            'pose_filled': state.pose_filled,
            'current_bin': bin_index,
            'is_complete': is_complete,
            'bbox': face.bbox.tolist(),
            'user_id': state.current_user_id
        }

# -------------------- ROUTES --------------------
@app.get("/", response_class=HTMLResponse)
async def index():
    try:
        if os.path.exists("index.html"):
            with open("index.html", "r") as f:
                return f.read()
        else:
            return "<h1>Error: index.html not found</h1>"
    except Exception as e:
        return f"<h1>Error loading page: {str(e)}</h1>"

@app.post("/start_enrollment")
async def start_enrollment(request: EnrollmentRequest = None):
    try:
        if db is None:
            raise HTTPException(status_code=500, detail="Firebase not initialized")
        
        init_face_model()
        
        user_id = request.user_id if request and request.user_id else None
        user_name = request.user_name if request and request.user_name else None
        user_email = request.user_email if request and request.user_email else None
        metadata = request.metadata if request and request.metadata else None
        
        state.reset(user_id, user_name, user_email, metadata)
        
        return {
            "status": "success",
            "message": "Enrollment started",
            "user_id": state.current_user_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process_frame")
async def process_frame(frame_data: FrameData):
    try:
        if not state.is_enrolling:
            return {"status": "error", "message": "Not enrolling"}
        
        image_data = frame_data.image.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        result = process_enrollment_frame(frame)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get_status")
async def get_status():
    with state.lock:
        filled_count = sum(state.pose_filled)
        progress = (filled_count / state.POSE_BINS) * 100
        return {
            "status": "success",
            "progress": progress,
            "pose_filled": state.pose_filled,
            "is_complete": filled_count == state.POSE_BINS,
            "message": state.last_message,
            "user_id": state.current_user_id
        }

@app.get("/get_user/{user_id}")
async def get_user(user_id: str):
    try:
        user_data = get_embedding_from_firestore(user_id)
        if user_data:
            return {
                "status": "success",
                "user_id": user_data['user_id'],
                "user_name": user_data.get('user_name'),
                "user_email": user_data.get('user_email'),
                "samples_count": user_data.get('samples_count'),
                "created_at": user_data.get('created_at'),
                "updated_at": user_data.get('updated_at')
            }
        else:
            return {"status": "error", "message": "User not found"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get_all_users")
async def get_all_users():
    try:
        users = get_all_embeddings()
        return {
            "status": "success",
            "users": users,
            "count": len(users)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search_face")
async def search_face(frame_data: FrameData):
    try:
        if state.face_app is None:
            init_face_model()
        
        image_data = frame_data.image.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        faces = state.face_app.get(frame)
        
        if len(faces) == 0:
            return {"status": "no_face", "message": "No face detected"}
        
        if len(faces) > 1:
            return {"status": "multiple_faces", "message": "Multiple faces detected"}
        
        face = faces[0]
        emb = l2_normalize(face.embedding)
        
        matches = search_similar_embeddings(emb, threshold=0.6, limit=5)
        
        return {
            "status": "success",
            "matches": matches,
            "count": len(matches)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
