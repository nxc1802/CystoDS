-- Layout filter for the Vietnamese CystoDS manuscript.
-- Pass-through Lua filter as table and figure floats are processed in build_paper_pdf.py

function Image(img)
  return img
end
