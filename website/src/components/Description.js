import React, { useState, useEffect, useRef } from "react";
import Typist from 'react-typist';
import LazyLoad from 'react-lazyload';
import { CopyToClipboard } from 'react-copy-to-clipboard';

// reactstrap components
import {
  Button,
  Container,
  Row,
  Col,
  Nav,
  NavItem,
  NavLink,
  Tooltip,
  UncontrolledPopover,
  PopoverHeader,
  PopoverBody
} from "reactstrap";

export default function Basics() {
  const [copied, setCopied] = useState(false);
  const [tooltipOpen, setTooltipOpen] = useState(false);
  const refBibTex = useRef();
  const toggle = () => {
    setTooltipOpen(!tooltipOpen);
    setCopied(false);
  };
  const TooltipContent = ({ scheduleUpdate, copied }) => {
    useEffect(scheduleUpdate, [copied, scheduleUpdate])
    return (
      <>{ copied ? "Copied!" : "Copy to Clipboard" }</>
    );
  }
  return (
    <div className="section section-basic" id="basic-elements">
      <img
        alt="..."
        className="path"
        src={require("assets/img/path1.png").default}
      />
      <Container>
        <Row>
          <Col md="12">
            <h1 className="title">Most Text-Based ML Systems Are Broken.</h1>
            <p className="main-text">Unlike human writing, modern computers can encode any given piece of text with a near-infinite number of unique logical encodings. This is derived from the fact that common language encodings, such as <a href="https://unicode.org" target="_blank" rel="noreferrer">Unicode</a>, provide methods to create differences between the logical encoding of a string and its visual rendering.</p>
            <p className="main-text pt-3 pb-5">Since text-based ML models, and most NLP systems more broadly, operate upon the logical encoding of text as inputs, the difference between logical encoding and visual rendering can be used to deceive users and adversarially control the output of these systems.</p>
            <LazyLoad><h1 className="text-center"><code><Typist stdTypingDelay={50} avgTypingDelay={100}>Send money to 1234 --> Send money to 2314</Typist></code></h1></LazyLoad>

            <h1 className="title pt-5 mt-5">These Differences Can Manipulate NLP Systems.</h1>
            <p className="main-text">Machine translation systems, search engines, spam filters, text classification, and nearly any other system which processes raw text-based user input can be manipulated using these tactics, which we collectively label imperceptible perturbations.</p>
            <p className="main-text pt-3">Exploitation of differences between logical and visual representations of text can take different forms in different settings.</p>
            <p className="main-text pt-3">Machine learning-based NLP systems are vulnerable to adversarial perturbations that are imperceptible to human users. This means that a motivated adversary can use imperceptible perturbations to control a system's output for a fixed visual input. Similarly, such perturbations can be used to poison training data.</p>
            <p className="main-text pt-3">These same methods can be used in an entirely different setting not just to degrade the performance of search engines, but also to allow content publishers to functionally hide content from search engine indexing systems.</p>

            <h1 className="title pt-5">There Are Four Flavors of Attack.</h1>
            <p className="main-text">Four different classifications of imperceptible perturbations exist:</p>
            <Row className="main-text text-center mt-4">
              <Col className="mb-4 mb-md-0">
                <Nav className="nav-pills-icons nav-pills-info justify-content-around" pills>
                  <NavItem id="invisibles" className="min-width-10 pt-4">
                    <NavLink className="active bg-img-none">
                      <i className="tim-icons icon-light-3" />
                      Invisible Chars
                    </NavLink>
                  </NavItem>
                  <UncontrolledPopover target="invisibles" placement="top" trigger="hover" fade={false}>
                    <PopoverHeader>Invisible Characters</PopoverHeader>
                    <PopoverBody>
                    Invisible Characters are a subset of Unicode characters that are not intended to render to a visible glyph, such as <i>zero width spaces</i>. These cross-platform characters can be injected into strings with no limit.
                    </PopoverBody>
                  </UncontrolledPopover>
                  <NavItem id="homoglyphs" className="min-width-10 pt-4">
                    <NavLink className="active bg-img-none">
                      <i className="tim-icons icon-single-copy-04" />
                      Homoglyphs
                    </NavLink>
                  </NavItem>
                  <UncontrolledPopover target="homoglyphs" placement="top" trigger="hover" fade={false}>
                    <PopoverHeader>Homoglyphs</PopoverHeader>
                    <PopoverBody>
                      Homoglyphs are distinct characters that render to the same or nearly the same glyph, such as the Latin <code>a</code> and the Cyrillic <code>а</code>. If any homoglyphs exist for a certain character, they can be swapped freely in most fonts.
                    </PopoverBody>
                  </UncontrolledPopover>
                  <NavItem id="reorderings" className="min-width-10 pt-4">
                    <NavLink className="active bg-img-none">
                      <i className="tim-icons icon-refresh-02" />
                      Reorderings
                    </NavLink>
                  </NavItem>
                  <UncontrolledPopover target="reorderings" placement="top" trigger="hover" fade={false}>
                    <PopoverHeader>Reorderings</PopoverHeader>
                    <PopoverBody>
                      Reorderings are methods by which special control characters can be used to change the rendering order of encoded characters. Although rendering order implementations vary by platform, well-crafted reorderings will render as desired on most modern platforms and can be injected an arbitrary number of times.
                    </PopoverBody>
                  </UncontrolledPopover>
                  <NavItem id="deletions" className="min-width-10 pt-4">
                    <NavLink className="active bg-img-none">
                      <i className="tim-icons icon-simple-remove" />
                      Deletions
                    </NavLink>
                  </NavItem>
                  <UncontrolledPopover target="deletions" placement="top" trigger="hover" fade={false}>
                    <PopoverHeader>Deletions</PopoverHeader>
                    <PopoverBody>
                      Deletions are methods by which control characters designed to remove text, such as <code>backspace</code>, are used to hide characters within strings. Deletions are platform dependent and will only render as desired in some settings, such as strings passed through Python's <code>print()</code> function.
                    </PopoverBody>
                  </UncontrolledPopover>
                </Nav>
              </Col>
            </Row>
            <p className="main-text pt-4 mt-4"><b>Invisible Characters</b> are a subset of characters that are not intended to render to a visible glyph, such as <i>zero width spaces</i>. These cross-platform characters can be injected into strings with no limit.</p>
            <p className="main-text pt-3"><b>Homoglyphs</b> are distinct characters that render to the same or nearly the same glyph, such a the Latin <code>a</code> and the Cyrillic <code>а</code>. If any homoglyphs exist for a certain character, they can be swapped freely in most fonts.</p>
            <p className="main-text pt-3"><b>Reorderings</b> are methods by which special control characters can be used to change the rendering order of encoded characters. Although rendering order implementations vary by platform, well-crafted reorderings will render as desired on most modern platforms and can be injected an arbitrary number of times.</p>
            <p className="main-text pt-3"><b>Deletions</b> are methods by which control characters designed to remove text, such as <code>backspace</code>, are used to hide characters within strings. Deletions are platform depdendent and will only render as desired in some settings, such as strings passed through Python's <code>print()</code> function.</p>

            <h1 className="title pt-5">Defenses Exist.</h1>
            <p className="main-text">It's possible to defend against imperceptible perturbation attacks.</p>
            <p className="main-text pt-3">These defenses take different forms in different settings and can be quite nuanced. The proper defense for one setting, such as English language NLP systems, may not be appropriate for other settings, such as multilingual search engines.</p>
            <p className="main-text pt-3">The key defense takeaway for imperceptible perturbations is that user inputs must be sanitized before ingress into an NLP pipeline. Without this, users may be vulnerable to adversarially manipulated results. Much like the consequences of SQL injection, imperceptible perturbations require conscious design decisions for all systems using affected technologies.</p>

            <h1 className="title pt-5">Try It Out.</h1>
            <p className="main-text">Use the <a href="/generator">Perturbation Generator</a> tool to generate your own imperceptible perturbations in the browser. If you'd like to check whether a specific string contains imperceptible perturbations, just paste it into the <a href="/detector">Attack Detector</a> tool.</p>


            <h1 className="title pt-5">There's More to Know.</h1>
            <p className="main-text">Read our <a href="https://arxiv.org/pdf/2106.09898.pdf">paper</a> to learn the details of crafting and defending against imperceptible perturbations.</p>
            <p className="main-text pt-3 pb-4">If you use our paper or anything on this site in your own work, please cite the following:</p>
            <div className="bibtex d-flex flex-wrap">
              <div ref={refBibTex}>
                <span className="code">@article&#123;boucher_imperceptible_2021,</span>
                  <span className="code tab">title = &#123;Bad &#123;Characters&#125;: &#123;Imperceptible&#125; &#123;NLP&#125; &#123;Attacks&#125;&#125;,</span>
                  <span className="code tab">author = &#123;Nicholas Boucher and Ilia Shumailov and Ross Anderson and Nicolas Papernot&#125;,</span>
                  <span className="code tab">year = &#123;2021&#125;,</span>
                  <span className="code tab">journal = &#123;Preprint&#125;,</span>
                  <span className="code tab">eprint = &#123;2106.09898&#125;,</span>
                  <span className="code tab">archivePrefix = &#123;arXiv&#125;,</span>
                  <span className="code tab">primaryClass = &#123;cs.CL&#125;,</span>
                  <span className="code tab">url = &#123;https://arxiv.org/abs/2106.09898&#125;</span>
                <span className="code">&#125;</span>
              </div>
              <div className="ml-auto align-self-end">
                  <CopyToClipboard text={refBibTex?.current?.innerText} onCopy={(text,result) => setCopied(result) }>
                    <Button id="copy" className="btn-round btn-icon ml-auto" color="default">
                        <i className="tim-icons icon-single-copy-04" />
                    </Button>
                  </CopyToClipboard>
                  <Tooltip delay={0} target="copy" isOpen={tooltipOpen} toggle={toggle}>
                    {({scheduleUpdate}) => (<TooltipContent copied={copied} scheduleUpdate={scheduleUpdate} />)}
                  </Tooltip>
                </div>
            </div>
          </Col>
        </Row>
      </Container>
    </div>
  );
}