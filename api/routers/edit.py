"""
api/routers/edit.py
Document editing endpoints.
"""
import os
import sys
sys.path.insert(0, os.path.abspath('.'))

from fastapi import APIRouter, HTTPException
from api.models import EditRequest, EditResponse

"""
api/routers/edit.py
Document editing endpoints.
"""
import os
import sys
sys.path.insert(0, os.path.abspath('.'))

from fastapi import (
    APIRouter, UploadFile, File, HTTPException
)
from api.models import EditRequest, EditResponse

router = APIRouter()    # ← only ONE of these in the whole file

EDIT_UPLOAD_DIR = "data/test_docs"
os.makedirs(EDIT_UPLOAD_DIR, exist_ok=True)


@router.post("/edit", response_model=EditResponse)
async def edit_document(request: EditRequest):
    try:
        from agents.editor_agent import run_editor

        full_instruction = (
            f"{request.instruction} "
            f"File: {request.file_path}"
        )
        result = run_editor(full_instruction, verbose=False)

        return EditResponse(
            instruction=request.instruction,
            result=result["result"],
            success=True
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Edit failed: {str(e)}"
        )


@router.post("/edit/upload")
async def upload_file_for_editing(
    file: UploadFile = File(...)
):
    save_path = f"{EDIT_UPLOAD_DIR}/{file.filename}"
    content   = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    return {
        "filename": file.filename,
        "path":     save_path,
        "size_kb":  len(content) // 1024
    }


@router.get("/edit/tools")
async def list_tools():
    from agents.editor_tools.tool_registry import (
        ALL_EDITOR_TOOLS
    )
    return {
        "tools": [
            {"name": t.name,
             "description": t.description.split("\n")[0]}
            for t in ALL_EDITOR_TOOLS
        ],
        "total": len(ALL_EDITOR_TOOLS)
    }

# router = APIRouter()

from fastapi import UploadFile, File
 
EDIT_UPLOAD_DIR = "data/test_docs"
os.makedirs(EDIT_UPLOAD_DIR, exist_ok=True)
 
 
@router.post("/edit/upload")
async def upload_file_for_editing(
    file: UploadFile = File(...)
):
    """
    Upload a file to the server so the Editor Agent
    can operate on it. Returns the server-side path
    to use in the subsequent /edit call.
    """
    save_path = f"{EDIT_UPLOAD_DIR}/{file.filename}"
    content   = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)
 
    return {
        "filename": file.filename,
        "path":     save_path,
        "size_kb":  len(content) // 1024
    }


# @router.post("/edit",
#              response_model=EditResponse)
# async def edit_document(request: EditRequest):
#     """
#     Edit a document using natural language.
#     Supports .docx, .pdf, and image files.

#     Example instructions:
#     - Replace DRAFT with FINAL in report.docx
#     - Add CONFIDENTIAL watermark to contract.pdf
#     - Resize logo.png to 800px width
#     """
#     try:
#         from agents.editor_agent import run_editor

#         full_instruction = (
#             f"{request.instruction} "
#             f"File: {request.file_path}"
#         )

#         result = run_editor(
#             full_instruction,
#             verbose=False
#         )

#         return EditResponse(
#             instruction=request.instruction,
#             result=result["result"],
#             success=True
#         )

#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail=f"Edit failed: {str(e)}"
#         )


@router.get("/edit/tools")
async def list_tools():
    """List all available editor tools."""
    from agents.editor_tools.tool_registry import (
        ALL_EDITOR_TOOLS
    )
    return {
        "tools": [
            {
                "name": t.name,
                "description": (
                    t.description.split("\n")[0]
                )
            }
            for t in ALL_EDITOR_TOOLS
        ],
        "total": len(ALL_EDITOR_TOOLS)
    }