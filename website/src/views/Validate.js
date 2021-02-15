import {useState, useEffect} from "react";

// core components
import IndexNavbar from "components/IndexNavbar.js";
import Footer from "components/Footer.js";

// reactstrap components
import {
  Container,
  Row,
  Col,
  Form,
  FormGroup,
  Label,
  Input,
  Nav,
  NavItem,
  NavLink,
  UncontrolledPopover,
  PopoverHeader,
  PopoverBody
} from "reactstrap";

import {invisibleChars, reorderingChars, deletionChars} from "variables/Constants.js";
import unicodeRanges from "unicode-range-json";

const unicodeRange = (char) => {
  if (char.length === 1) {
    for (let i=0; i<unicodeRanges.length; i++) {
      const codePoint = char.codePointAt(0);
      if (codePoint >= unicodeRanges[i].range[0] && codePoint <= unicodeRanges[i].range[1]) {
        return unicodeRanges[i].category;
      }
    }
  }
};

export default function Validate() {
  const [input, setInput] = useState("");
  const [output, setOutput] = useState("");
  const [invisibles, setInvisibles] = useState(false);
  const [homoglyphs, setHomoglyphs] = useState(false);
  const [reorderings, setReorderings] = useState(false);
  const [deletions, setDeletions] = useState(false);
  useEffect(() => {
    // Reset all filters
    setInvisibles(false);
    setHomoglyphs(false);
    setReorderings(false);
    setDeletions(false);
    // Test for invisible characters
    for (let i=0; i<invisibleChars.length; i++) {
      if (input.includes(invisibleChars[i])) {
        setInvisibles(true);
        break;
      }
    }
    //Test for homoglyphs
    const tokens = input.split(/[\s,\d'"“”‘’\-+%*$#@!?°=_/\\.()^$\xA2-\xA5\u058F\u060B\u09F2\u09F3\u09FB\u0AF1\u0BF9\u0E3F\u17DB\u20A0-\u20BD\uA838\uFDFC\uFE69\uFF04\uFFE0\uFFE1\uFFE5\uFFE6]+/);
    loop: for (let i=0; i<tokens.length; i++) {
      if (tokens[i].length) {
        let block = '';
        for (let j=0; j<tokens[i].length; j++) {
          let newblock = unicodeRange(tokens[i][j]);
          if (newblock !== "General Punctuation" && newblock !== "Control Character") {
            if (block && newblock !== block) {
              setHomoglyphs(true);
              break loop;
            }
            else {
              block = newblock;
            }
          }
        }
      }
    }
    // Test for reordering characters
    for (let i=0; i<reorderingChars.length; i++) {
      if (input.includes(reorderingChars[i])) {
        setReorderings(true);
        break;
      }
    }
    // Test for deletion characters
    for (let i=0; i<deletionChars.length; i++) {
      if (input.includes(deletionChars[i])) {
        setDeletions(true);
        break;
      }
    }
    // Update output visualization
    setOutput(input.split("\n").map((p, i) => <p className="pt-2" key={`p${i}`}>{p.split("").map((char, j) => {
      const codePoint = char.codePointAt(0);
      if (codePoint >= 0x20 && codePoint <= 0x7E) {
        return char;
      }
      else {
        const code = `U+${codePoint.toString(16).toUpperCase()}`;
        return <>
                  <span className="unicode" id={`pop-${i}-${j}`}>{code}</span>
                  <UncontrolledPopover target={`pop-${i}-${j}`} placement="top" trigger="hover" fade={false}>
                    <PopoverHeader>{code}</PopoverHeader>
                    <PopoverBody className="text-center">
                      <h2 className="text-info">{char}</h2>
                      Unicode Range: {unicodeRange(char)}
                    </PopoverBody>
                  </UncontrolledPopover>
               </>;
      }
    })}</p>));
  }, [input]);
  return (
    <>
      <IndexNavbar />
      <div className="wrapper">
        <div className="page-header d-flex flex-wrap">
          <img
            alt="..."
            className="dots"
            src={require("assets/img/dots.png").default}
          />
          <img
            alt="..."
            className="path"
            src={require("assets/img/path2.png").default}
          />
          <div className="w-100">
            <Container className="below-nav mb-4">
                <Row>
                    <Col md={12} className="text-center">
                      <h1><b>Validate</b></h1>
                    </Col>
                </Row>
                <Form>
                  <FormGroup row className="mt-5 mb-n2">
                      <Label for="inputText" md={2}><h2>Input&nbsp;={'>'}</h2></Label>
                      <Col md={10}>
                          <Input
                              type="textarea"
                              name="inputText"
                              id="inputText"
                              placeholder="Paste some text here to validate whether it has been manipulated with imperceptible perturbations..."
                              onChange={e => setInput(e.target.value)}
                          />
                      </Col>
                  </FormGroup>
              </Form>
              <Row className="mt-4 pt-4 mb-4 pb-4">
                  <Col md={2} style={{ visibility: input.length ? null : 'hidden' }}>
                      <h3>Output&nbsp;={'>'}</h3>
                  </Col>
                  <Col md={10}>
                    <Row className="pb-4">
                      <Col md={3} xs={6} className="mb-4 mb-md-0">
                        <Nav className="nav-pills-icons nav-pills-warning" pills>
                          <NavItem id="invisibles" className="min-width-9">
                            <NavLink className={invisibles ? "active" : null}>
                              <i className="tim-icons icon-light-3" />
                              Invisibles
                            </NavLink>
                          </NavItem>
                          <UncontrolledPopover target="invisibles" placement="top" trigger="hover" fade={false}>
                            <PopoverHeader>Invisible Characters</PopoverHeader>
                            <PopoverBody>
                              {invisibles ?
                                <span>This text <b>may contain</b> invisible characters.</span> :
                                <span>This text <b>likely doesn't contain</b> invisible characters.</span>}
                            </PopoverBody>
                          </UncontrolledPopover>
                        </Nav>
                      </Col>
                      <Col md={3} xs={6} className="mb-4 mb-md-0">
                        <Nav className="nav-pills-icons nav-pills-warning" pills>
                          <NavItem id="homoglyphs" className="min-width-9">
                            <NavLink className={homoglyphs ? "active" : null}>
                              <i className="tim-icons icon-single-copy-04" />
                              Homoglyphs
                            </NavLink>
                          </NavItem>
                          <UncontrolledPopover target="homoglyphs" placement="top" trigger="hover" fade={false}>
                            <PopoverHeader>Homoglyphs</PopoverHeader>
                            <PopoverBody>
                              {homoglyphs ?
                                <span>This text <b>may contain</b> homoglyphs.</span> :
                                <span>This text <b>likely doesn't contain</b> homoglyphs.</span>}
                            </PopoverBody>
                          </UncontrolledPopover>
                        </Nav>
                      </Col>
                      <Col md={3} xs={6} className="mb-4 mb-md-0">
                        <Nav className="nav-pills-icons nav-pills-warning" pills>
                          <NavItem id="reorderings" className="min-width-9">
                            <NavLink className={reorderings ? "active" : null}>
                              <i className="tim-icons icon-refresh-02" />
                              Reorderings
                            </NavLink>
                          </NavItem>
                          <UncontrolledPopover target="reorderings" placement="top" trigger="hover" fade={false}>
                            <PopoverHeader>Reorderings</PopoverHeader>
                            <PopoverBody>
                              {reorderings ?
                                <span>This text <b>may contain</b> reorderings.</span> :
                                <span>This text <b>likely doesn't contain</b> reorderings.</span>}
                            </PopoverBody>
                          </UncontrolledPopover>
                        </Nav>
                      </Col>
                      <Col md={3} xs={6} className="mb-4 mb-md-0">
                        <Nav className="nav-pills-icons nav-pills-warning" pills>
                          <NavItem id="deletions" className="min-width-9">
                            <NavLink className={deletions ? "active" : null}>
                              <i className="tim-icons icon-simple-remove" />
                              Deletions
                            </NavLink>
                          </NavItem>
                          <UncontrolledPopover target="deletions" placement="top" trigger="hover" fade={false}>
                            <PopoverHeader>Deletions</PopoverHeader>
                            <PopoverBody>
                              {deletions ?
                                <span>This text <b>may contain</b> deletions.</span> :
                                <span>This text <b>likely doesn't contain</b> deletions.</span>}
                            </PopoverBody>
                          </UncontrolledPopover>
                        </Nav>
                      </Col>
                    </Row>
                    <div className="validated mt-4" style={{ display: input.length ? null : 'None' }}>
                      <p className="text-muted"><b>Encoding Visualization:</b></p>
                      {output}
                    </div>
                  </Col>
              </Row>
            </Container>
          </div>
          <div className="w-100 align-self-end">
            <Container className="pt-0 pb-4">
                <Row>
                    <Col md={{ size: 10, offset: 2}}>
                      <p className="text-muted">This tool tests whether the input string contains encodings that may be indicators of imperceptible perturbations. It is not guaranteed to detect all forms of imperceptible perturbations. All text entered remains on your local machine. Nothing is transmitted to or logged on any server. This tool is for academic purposes only and the user holds sole responsibility for how it is used.</p>
                    </Col>
                </Row>
            </Container>
          </div>
      </div>
      <Footer />
    </div>
    </>
  );
}
